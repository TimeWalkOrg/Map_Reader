"""U-Net architecture shared by train.py (self-contained copy) and infer_vectorize.py."""
import torch, torch.nn as nn, torch.nn.functional as F


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


