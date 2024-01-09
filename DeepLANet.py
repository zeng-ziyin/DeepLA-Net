import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import vector_feature

def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)

class VFR(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.linear(x)
        x = vector_feature(x, knn, self.training)
        x = self.bn(x.view(B*N, -1)).view(B, N, -1)
        return x

class FFN(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.ffn[-1].weight, init)

    def forward(self, x):
        B, N, C = x.shape
        x = self.ffn(x.view(B*N, -1)).view(B, N, -1)
        return x

class ResLFE_Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()

        self.depth = depth
        self.VFRs = nn.ModuleList([VFR(dim, dim, bn_momentum) for _ in range(depth)])
        self.mlp = FFN(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.FFNs = nn.ModuleList([FFN(dim, mlp_ratio, bn_momentum, act) for _ in range(depth)])

        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        self.drop_paths = nn.ModuleList([DropPath(dpr) for dpr in drop_rates])

    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, pe, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0, pts)
        for i in range(self.depth):
            x = x + pe
            x = x + self.drop_path(self.VFRs[i](x, knn), i, pts)
            x = x + self.drop_path(self.FFNs[i](x), i, pts)
        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth
        self.up_depth = len(args.depths) - 1
        self.first = first = depth == 0
        self.last = last = depth == self.up_depth
        self.k = args.ks[depth]
        self.cp = cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        dim = args.dims[depth]
        if first:
            nbr_in_dim = 7
            nbr_hid_dim = args.nbr_dims[0]
            nbr_out_dim = dim
            self.nbr_embed = nn.Sequential(
                nn.Linear(nbr_in_dim, nbr_hid_dim // 2, bias=False),
                nn.BatchNorm1d(nbr_hid_dim // 2, momentum=cp_bn_momentum),
                args.act(),
                nn.Linear(nbr_hid_dim // 2, nbr_hid_dim, bias=False),
                nn.BatchNorm1d(nbr_hid_dim, momentum=cp_bn_momentum),
                args.act(),
                nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False),
            )
            self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            nn.init.constant_(self.nbr_bn.weight, 0.8)
            self.nbr_proj = nn.Identity()

        pe_in_dim = 3
        pe_hid_dim = args.nbr_dims[1] // 2
        pe_out_dim = args.nbr_dims[1]
        self.pe_embed = nn.Sequential(
            nn.Linear(pe_in_dim, pe_hid_dim//2, bias=False),
            nn.BatchNorm1d(pe_hid_dim//2, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(pe_hid_dim//2, pe_hid_dim, bias=False),
            nn.BatchNorm1d(pe_hid_dim, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(pe_hid_dim, pe_out_dim, bias=False),
        )
        self.pe_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.pe_bn.weight, 0.2)
        self.pe_proj = nn.Linear(pe_out_dim, dim, bias=False)

        if not first:
            in_dim = args.dims[depth - 1]
            self.vfr = VFR(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.reslfe = ResLFE_Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act)
        self.drop = DropPath(args.head_drops[depth])

        self.sem_sup = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.num_classes, bias=False),
        )
        nn.init.constant_(self.sem_sup[0].weight, (args.dims[0] / dim) ** 0.5)

        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)

    def local_aggregation(self, x, pe, knn, pts):
        x = x.unsqueeze(0)
        x = self.reslfe(x, pe, knn, pts)
        x = x.squeeze(0)
        return x

    def forward(self, x, xyz, prev_knn, indices, pts_list):
        """
        x: N x C
        """
        # downsampling
        if not self.first:
            ids = indices.pop()
            xyz = xyz[ids]
            x = self.skip_proj(x)[ids] + self.vfr(x.unsqueeze(0), prev_knn).squeeze(0)[ids]
        knn = indices.pop()

        # spatial encoding
        N, k = knn.shape
        pe = xyz[knn] - xyz.unsqueeze(1)

        if self.first:
            nbr = pe.clone()
            nbr = torch.cat([nbr, x[knn]], dim=-1).view(-1, 7)
            if self.training and self.cp:
                nbr.requires_grad_()
            nbr_embed_func = lambda x: self.nbr_embed(x).view(N, k, -1).max(dim=1)[0]
            nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
            nbr = self.nbr_proj(nbr)
            nbr = self.nbr_bn(nbr)
            x = nbr

        pe = pe.view(-1, 3)
        if self.training and self.cp:
            pe.requires_grad_()
        pe_embed_func = lambda x: self.pe_embed(x).view(N, k, -1).max(dim=1)[0]
        pe = checkpoint(pe_embed_func, pe) if self.training and self.cp else pe_embed_func(pe)
        pe = self.pe_proj(pe)
        pe = self.pe_bn(pe)

        # main block
        knn = knn.unsqueeze(0)
        pts = pts_list.pop() if pts_list is not None else None
        x = checkpoint(self.local_aggregation, x, pe, knn, pts) if self.training and self.cp else self.local_aggregation(x, pe, knn, pts)

        # get subsequent feature maps
        if not self.last:
            sub_x, sub_spa, sub_sem = self.sub_stage(x, xyz, knn, indices, pts_list)
        else:
            sub_x = sub_spa = None
            sub_sem = []

        # Deep Supervision
        if self.training:
            rel_cor = torch.max(xyz[knn.squeeze(0)] - xyz.unsqueeze(1), dim=1, keepdim=False)[0]
            rel_cor.mul_(self.cor_std)
            rel_p = torch.max(x[knn.squeeze(0)] - x.unsqueeze(1), dim=1, keepdim=False)[0]
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_spa = sub_spa + closs if sub_spa is not None else closs

        sem = self.sem_sup(x)
        sub_sem = sub_sem.append(sem) if sub_sem is not None else [sem]

        # upsampling
        x = self.postproj(x)
        if not self.first:
            back_nn = indices[self.depth-1]
            x = x[back_nn]
        x = self.drop(x)
        sub_x = sub_x + x if sub_x is not None else x

        return sub_x, sub_spa, sub_sem

class DeepLA_semseg(nn.Module):
    def __init__(self, args):
        super().__init__()

        # bn momentum for checkpointed layers
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5
        self.stage = Stage(args)
        hid_dim = args.head_dim
        out_dim = args.num_classes

        self.seg_head = nn.Sequential(
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(hid_dim, hid_dim//2),
            nn.BatchNorm1d(hid_dim//2, momentum=args.bn_momentum),
            args.act(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim//2, out_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None):
        indices = indices[:]
        x, spa, sem = self.stage(x, xyz, None, indices, pts_list)
        if self.training:
            return self.seg_head(x), spa, sem
        return self.seg_head(x)

