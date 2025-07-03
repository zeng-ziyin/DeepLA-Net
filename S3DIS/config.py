from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# S3DIS dataset path
raw_data_path = Path("/media/zengziyin/ZZY/Zeng/DeepLA-Net/Data/Stanford3dDataset_v1.2_Aligned_Version")

processed_data_path = raw_data_path.parent / "s3dis"
# if you want to set the processed dataset path, uncomment here
#processed_data_path = Path("")

epoch = 100
warmup = 10
batch_size = 8
learning_rate = 4e-3
label_smoothing = 0.2

s3dis_args = SimpleNamespace()
s3dis_args.k = [24, 24, 24, 24]
s3dis_args.grid_size = [0.04, 0.08, 0.16, 0.32]

s3dis_args.max_pts = 30000

s3dis_warmup_args = deepcopy(s3dis_args)
s3dis_warmup_args.grid_size = [0.04, 3.5, 3.5, 3.5]

deepla_args = SimpleNamespace()
deepla_args.ks = s3dis_args.k
deepla_args.depths = [20, 20, 60, 20]  # DeepLA-120
# deepla_args.depths = [10, 10, 30, 10]  # DeepLA-60
deepla_args.dims = [64, 128, 256, 512]
deepla_args.nbr_dims = [32, 32]
deepla_args.head_dim = 256
deepla_args.num_classes = 13
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(deepla_args.depths)).split(deepla_args.depths)
deepla_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
deepla_args.head_drops = torch.linspace(0., 0.15, len(dela_args.depths)).tolist()
deepla_args.bn_momentum = 0.02
deepla_args.act = nn.GELU
deepla_args.mlp_ratio = 1
# gradient checkpoint
deepla_args.use_cp = False

deepla_args.cor_std = [1.6, 3.2, 6.4, 12.8]
