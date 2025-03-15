import random, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys, math
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from deepla_semseg import DeepLA_semseg
from time import time, sleep
from config import s3dis_args, s3dis_warmup_args, deepla_args, batch_size, learning_rate as lr, epoch, warmup, label_smoothing as ls

torch.set_float32_matmul_precision("high")

def warmup_fn(model, dataset):
    model.train()
    traindlr = DataLoader(dataset, batch_size=len(dataset), collate_fn=s3dis_collate_fn, pin_memory=True, num_workers=6)
    for xyz, feature, indices, pts, y in traindlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            p, closs, s_list = model(xyz, feature, indices, pts)
            sem_Loss = sem_sup(s_list, y, indices)
            loss = F.cross_entropy(p, y) + closs + sem_Loss
        loss.backward()

def sem_sup(sem_list, Y, indices):
    sem_Loss = 0
    for i in range(len(sem_list)):
        pred = sem_list[i]
        # pred = sem_list.pop()
        idx = indices[-2*i]
        y = Y[idx]
        # print(pred.shape, y.shape)
        loss = F.cross_entropy(pred, y, label_smoothing=ls)
        sem_Loss += loss
    sem_Loss *= .25
    return sem_Loss

cur_id = "01"
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
errfile = f"output/log/{cur_id}/err.log"
logfile = open(logfile, "a", 1)
errfile = open(errfile, "a", 1)
sys.stdout = logfile
sys.stderr = errfile
# 
print(r"base ")

traindlr = DataLoader(S3DIS(s3dis_args, partition="!5", loop=25), batch_size=batch_size,
                      collate_fn=s3dis_collate_fn, shuffle=True, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=16)
testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=1, train=False), batch_size=1,
                      collate_fn=s3dis_collate_fn, pin_memory=True, 
                      persistent_workers=True, num_workers=16)

step_per_epoch = len(traindlr)

model = DeepLA_semseg(deepla_args).cuda()

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial = epoch * step_per_epoch, lr_min = lr/10000,
                                warmup_t=warmup*step_per_epoch, warmup_lr_init = lr/20)
scaler = GradScaler()
# if wish to continue from a checkpoint
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(13)
ttls = util.AverageMeter() 
corls = util.AverageMeter()
semls = util.AverageMeter() 
best = 0
warmup_fn(model, S3DIS(s3dis_warmup_args, partition="!5", loop=batch_size, warmup=True))
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    corls.reset()
    metric.reset()
    semls.reset()
    now = time()
    for xyz, feature, indices, pts, y in traindlr:
        lam = scheduler_step/(epoch*step_per_epoch)
        alpha = 5e-3 ** lam * .25
        beta = 0.3 ** lam
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            p, spa_Loss, sem_list = model(xyz, feature, indices, pts)
            loss = F.cross_entropy(p, y, label_smoothing=ls)
            sem_Loss = sem_sup(sem_list, y, indices)
        metric.update(p.detach(), y)
        ttls.update(loss.item())
        corls.update(spa_Loss.item())
        semls.update(sem_Loss.item())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss + spa_Loss*alpha + sem_Loss*beta).backward()
        scaler.step(optimizer)
        scaler.update()
        
    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)} || sem: {round(semls.avg, 4)}")
    metric.print("train:")

    model.eval()
    metric.reset()
    with torch.no_grad():
        for xyz, feature, indices, pts, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            y = y.cuda(non_blocking=True)
            with autocast():
                p = model(xyz, feature, indices)
            metric.update(p, y)
    
    metric.print("val:  ")
    print(f"duration: {time() - now}")
    cur = metric.miou
    if best < cur:
        best = cur
        print("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)
    
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler, start_epoch=i+1)