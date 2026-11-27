"""Week 3 gate: does the small 4-block CNN beat the HFR baseline on blur size |dz| (same test split)?

Two targets, same network, normalisation and optimiser as train_check_v1.py:
  signed  - trained on dz, scored on |abs(prediction) - abs(dz)|
  size    - trained directly on |dz|
Each is trained with several seeds keeping the epoch with the
lowest error on the val split (nothing is selected on the test split).
Needs numpy and torch (GPU if available).  Usage: python cnn_vs_hfr.py [defocus_stars_v1.npz] [epochs] [seeds]"""
import sys, time
import numpy as np, torch, torch.nn as nn

from hfr_baseline import EDGES, fit_curve, hfr


def normalise(adu):
    """Border median / border std, as in train_check_v1.py."""
    a = adu.astype(np.float32)
    border = np.concatenate([a[:, :4].reshape(len(a), -1), a[:, -4:].reshape(len(a), -1),
                             a[:, :, :4].reshape(len(a), -1), a[:, :, -4:].reshape(len(a), -1)], 1)
    med = np.median(border, 1)[:, None, None]
    sd = border.std(1)[:, None, None] + 1e-6
    return ((a - med) / sd)[:, None] / 10


def make_net():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))


def train(X, y, tr, va, te, scale, epochs, seed, device):
    """Train for `epochs`, keep the epoch with the lowest val error, return test predictions and that epoch."""
    torch.manual_seed(seed)
    net = make_net().to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    target = torch.from_numpy(y / scale).to(device)
    def predict(ids):
        net.eval()
        with torch.no_grad():
            ids = torch.from_numpy(ids).to(device)
            return torch.cat([net(X[ids[i:i + 500]]).squeeze(1) for i in range(0, len(ids), 500)]).cpu().numpy() * scale

    best = None
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(np.random.default_rng(1000 * seed + ep).permutation(tr)).to(device)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            nn.functional.l1_loss(net(X[idx]).squeeze(1), target[idx]).backward()
            opt.step()
        val = np.abs(predict(va) - y[va]).mean()
        if best is None or val < best[0]:
            best = (val, ep + 1, predict(te))
    return best[2], best[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else "defocus_stars_v1.npz"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(path)
    adu, dz = z["adu"], z["dz_um"].astype(np.float32)
    tr, va, te = z["train_idx"], z["val_idx"], z["test_idx"]
    size = np.abs(dz)
    y = size[te]

    h = hfr(adu, 2.0, 38)      # variant chosen on the val split by hfr_baseline.py
    hfr_err = np.abs(np.interp(h[te], *fit_curve(h[tr], size[tr])) - y)

    X = torch.from_numpy(normalise(adu)).to(device)
    scale = float(size.max())
    errs = {}
    print(f"device: {device}, epochs: {epochs}, seeds: {seeds}")
    for name, target in (("signed", dz), ("size", size)):
        runs = []
        for seed in range(seeds):
            t = time.time()
            p, best_ep = train(X, target, tr, va, te, scale, epochs, seed, device)
            runs.append(np.abs(np.abs(p) - y))
            extra = ""
            if name == "signed":
                extra = (f", signed MAE {np.abs(p - dz[te]).mean():.1f} µm, "
                         f"sign acc {(np.sign(p) == np.sign(dz[te])).mean():.1%}")
            print(f"  {name} seed {seed}: |dz| MAE {runs[-1].mean():.1f} µm{extra}  (best epoch {best_ep}, {time.time() - t:.0f} s)",
                  flush=True)
        errs[name] = np.stack(runs)

    print(f"\ntest MAE of |dz| (µm), CNN = mean over {seeds} seeds")
    print("|dz| bin (µm)    n     HFR  CNN signed  CNN size")
    rows = [(f"{a}-{b}", (y >= a) & (y < b)) for a, b in zip(EDGES[:-1], EDGES[1:])]
    for label, m in rows + [("all", np.ones(len(y), bool))]:
        if m.any():
            print(f"{label:<13} {m.sum():>5} {hfr_err[m].mean():>7.1f} {errs['signed'][:, m].mean():>11.1f} "
                  f"{errs['size'][:, m].mean():>9.1f}")

    # paired bootstrap over test images: HFR error minus CNN error (positive = CNN better)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(y), (10000, len(y)))
    for name, e in errs.items():
        d = hfr_err - e.mean(0)
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        print(f"HFR - CNN {name}: {d.mean():+.1f} µm (95% CI {lo:+.1f} to {hi:+.1f})")


if __name__ == "__main__":
    main()
