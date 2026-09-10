#!/usr/bin/env python
"""Small U-Net trainer (pure torch + numpy; runs on pc5090 GPU or CPU).
Usage: python train.py --task roads|buildings --data nyc_tiles.npz --out ckpt_dir [--iters N] [--minutes M]
Loss on valid pixels only (weight map). Block-level spatial split (train tiles never contain val pixels)."""
import argparse, json, time, os, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["roads", "buildings"])
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--iters", type=int, default=6000)
p.add_argument("--minutes", type=float, default=20)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--crop", type=int, default=256)
p.add_argument("--base", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=1776)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)


class Block(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(s, x): return s.c(x)


class UNet(nn.Module):
    def __init__(s, cin=2, base=32, depth=5):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        s.enc = nn.ModuleList([Block(cin if i == 0 else ch[i - 1], ch[i]) for i in range(depth)])
        s.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth - 1, 0, -1)])
        s.dec = nn.ModuleList([Block(ch[i - 1] * 2, ch[i - 1]) for i in range(depth - 1, 0, -1)])
        s.head = nn.Conv2d(ch[0], 1, 1)
    def forward(s, x):
        skips = []
        for i, e in enumerate(s.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2)); skips.append(x)
        for u, d, sk in zip(s.up, s.dec, skips[-2::-1]):
            x = d(torch.cat([u(x), sk], 1))
        return s.head(x)


d = np.load(a.data)
img = d["img"]; split = d["split"]
tgt = d["road_tgt" if a.task == "roads" else "b_tgt"]; w = d["road_w" if a.task == "roads" else "b_w"]
tr_idx = np.where((split == 1) & (w.reshape(len(w), -1).sum(1) > 0))[0]
va_idx = np.where((split == 2) & (w.reshape(len(w), -1).sum(1) > 0))[0]
print(a.task, "train tiles", len(tr_idx), "val tiles", len(va_idx), "device", dev, flush=True)
pos_frac = float((tgt[tr_idx] * w[tr_idx]).sum() / max(1, w[tr_idx].sum()))
pos_weight = float(min(6.0, max(1.0, (1 - pos_frac) / max(pos_frac, 1e-3) * 0.5)))
print("pos_frac", pos_frac, "pos_weight", pos_weight, flush=True)

X = torch.from_numpy(img); Y = torch.from_numpy(tgt); Wt = torch.from_numpy(w)
Xv = X[va_idx].to(dev).float() / 255; Yv = Y[va_idx].to(dev).float(); Wv = Wt[va_idx].to(dev).float()


def augment(x, y, wm):
    """x float [B,2,H,W] 0..1; y,wm float [B,1,H,W]"""
    B = x.shape[0]
    # random crop
    H = x.shape[-1]; c = a.crop
    oy, ox = np.random.randint(0, H - c + 1), np.random.randint(0, H - c + 1)
    x, y, wm = x[..., oy:oy + c, ox:ox + c], y[..., oy:oy + c, ox:ox + c], wm[..., oy:oy + c, ox:ox + c]
    # scale jitter (whole batch)
    s = float(np.exp(np.random.uniform(np.log(0.8), np.log(1.25))))
    if abs(s - 1) > 0.03:
        n = int(round(c * s)) // 16 * 16
        x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(n, n), mode="nearest"); wm = F.interpolate(wm, size=(n, n), mode="nearest")
    # flips / rot90
    if np.random.rand() < 0.5: x, y, wm = x.flip(-1), y.flip(-1), wm.flip(-1)
    if np.random.rand() < 0.5: x, y, wm = x.flip(-2), y.flip(-2), wm.flip(-2)
    k = np.random.randint(4)
    if k: x, y, wm = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1)), torch.rot90(wm, k, (-2, -1))
    # intensity: gamma on ch0, stroke thicken/thin on ch1, noise, channel dropout
    g = torch.exp(torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.5, 0.5))
    x0 = x[:, :1].clamp(1e-4, 1) ** g
    x1 = x[:, 1:]
    r = np.random.rand()
    if r < 0.3: x1 = F.max_pool2d(x1, 3, 1, 1)
    elif r < 0.5: x1 = -F.max_pool2d(-x1, 3, 1, 1)
    x = torch.cat([x0, x1], 1)
    x = x + torch.randn_like(x) * 0.03 * np.random.rand()
    if np.random.rand() < 0.15:
        ch = np.random.randint(2); x[:, ch] = 0
    return x.clamp(0, 1), y, wm


def loss_fn(logit, y, wm):
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=torch.tensor(pos_weight, device=logit.device))
    bce = (bce * wm).sum() / wm.sum().clamp(min=1)
    pr = torch.sigmoid(logit) * wm; yy = y * wm
    dice = 1 - (2 * (pr * yy).sum() + 1) / (pr.sum() + yy.sum() + 1)
    return bce + 0.5 * dice


@torch.no_grad()
def evaluate(model):
    model.eval(); tp = fp = fn = 0.0; ls = 0.0; n = 0
    for i in range(0, len(Xv), 8):
        x, y, wm = Xv[i:i + 8], Yv[i:i + 8, None], Wv[i:i + 8, None]
        lg = model(x); ls += loss_fn(lg, y, wm).item() * len(x); n += len(x)
        pr = (lg > 0).float()
        tp += ((pr == 1) & (y == 1) & (wm == 1)).sum().item(); fp += ((pr == 1) & (y == 0) & (wm == 1)).sum().item()
        fn += ((pr == 0) & (y == 1) & (wm == 1)).sum().item()
    model.train()
    iou = tp / max(1, tp + fp + fn); prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return dict(val_loss=ls / max(1, n), val_iou=iou, val_precision=prec, val_recall=rec)


model = UNet(2, a.base).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
log = []; best = -1; t0 = time.time(); it = 0
while it < a.iters and (time.time() - t0) < a.minutes * 60:
    idx = np.random.choice(tr_idx, a.batch)
    x = X[idx].to(dev).float() / 255; y = Y[idx].to(dev).float()[:, None]; wm = Wt[idx].to(dev).float()[:, None]
    x, y, wm = augment(x, y, wm)
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        lg = model(x)
    loss = loss_fn(lg.float(), y, wm)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step(); it += 1
    if it % 250 == 0 or it == a.iters:
        ev = evaluate(model); ev.update(iter=it, train_loss=loss.item(), seconds=round(time.time() - t0, 1)); log.append(ev)
        print(json.dumps(ev), flush=True)
        if ev["val_iou"] > best:
            best = ev["val_iou"]; torch.save(model.state_dict(), f"{a.out}/{a.task}_best.pt")
torch.save(model.state_dict(), f"{a.out}/{a.task}_last.pt")
json.dump(dict(task=a.task, args=vars(a), device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
               torch=torch.__version__, n_train_tiles=int(len(tr_idx)), n_val_tiles=int(len(va_idx)), pos_frac=pos_frac,
               pos_weight=pos_weight, iters_done=it, seconds=round(time.time() - t0, 1), best_val_iou=best,
               n_params=sum(p.numel() for p in model.parameters()), log=log), open(f"{a.out}/{a.task}_train_log.json", "w"), indent=1)
print("done", a.task, "best val iou", best, "seconds", round(time.time() - t0, 1))
