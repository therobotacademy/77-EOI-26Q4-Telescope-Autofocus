"""Sanity check on dataset v1: a small CNN predicts signed dz (µm) from one 64x64 star.
Reports test MAE and sign accuracy per |dz| bin.  Usage: python train_check_v1.py [defocus_stars_v1.npz] [epochs]"""
import sys, time
import numpy as np, torch, torch.nn as nn

path = sys.argv[1] if len(sys.argv) > 1 else "defocus_stars_v1.npz"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
torch.manual_seed(0)
z = np.load(path)
adu = z["adu"].astype(np.float32)
# per-image normalisation: subtract the border median (sky + bias), divide by the border noise
border = np.concatenate([adu[:, :4].reshape(len(adu), -1), adu[:, -4:].reshape(len(adu), -1),
                         adu[:, :, :4].reshape(len(adu), -1), adu[:, :, -4:].reshape(len(adu), -1)], 1)
med = np.median(border, 1)[:, None, None]
sd = border.std(1)[:, None, None] + 1e-6
X = torch.from_numpy(((adu - med) / sd)[:, None] / 10)
dz = z["dz_um"].astype(np.float32)
scale = np.abs(dz).max()
Y = torch.from_numpy(dz / scale)
tr, te = z["train_idx"], z["test_idx"]

net = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), 1e-3)
t = time.time()
for ep in range(epochs):
    perm = torch.from_numpy(np.random.default_rng(ep).permutation(tr))
    net.train()
    for i in range(0, len(perm), 128):
        idx = perm[i:i + 128]
        opt.zero_grad()
        nn.functional.l1_loss(net(X[idx]).squeeze(1), Y[idx]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(X[te][i:i + 500]).squeeze(1) for i in range(0, len(te), 500)]).numpy() * scale
    y = dz[te]
    print(f"epoch {ep + 1}: test MAE {np.abs(p - y).mean():.0f} µm, sign acc {(np.sign(p) == np.sign(y)).mean():.1%}, "
          f"{time.time() - t:.0f} s", flush=True)
print("\n|dz| bin (µm)   n    MAE (µm)  sign acc")
edges = [0, 200, 400, 800, 1200, 1600]
for a, b in zip(edges[:-1], edges[1:]):
    m = (np.abs(y) >= a) & (np.abs(y) < b)
    print(f"{a:>5}-{b:<5} {m.sum():>5} {np.abs(p - y)[m].mean():>9.0f} {(np.sign(p) == np.sign(y))[m].mean():>9.1%}")
