"""Week 4: sign accuracy and error of the signed CNN per |dz| bin and across seeing, jitter and SNR.

Same network, normalisation and optimiser as cnn_vs_hfr.py. Training stops when the val MAE has not
improved for `patience` epochs; the best-val weights are saved and scored once on the test split.
Writes weights/signed_cnn_seed<k>.pt and weights/signed_test_predictions.npz.
Usage: python signed_robustness.py [defocus_stars_v1.npz] [max_epochs] [patience] [seeds]"""
import copy, os, sys, time
import numpy as np, torch, torch.nn as nn

from cnn_vs_hfr import make_net, normalise

OUT = "weights"
BINS = [0, 100, 200, 300, 400, 600, 800, 1211]
NEAR = (200, 400)          # candidate near-focus limits in µm


def predict(net, X, ids, scale):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(X[ids[i:i + 500]]).squeeze(1) for i in range(0, len(ids), 500)]).cpu().numpy() * scale


def train(X, dz, tr, va, scale, max_epochs, patience, seed, device):
    torch.manual_seed(seed)
    net = make_net().to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    target = torch.from_numpy(dz / scale).to(device)
    va_t = torch.from_numpy(va).to(device)
    best, best_ep, state = np.inf, 0, None
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = torch.from_numpy(np.random.default_rng(1000 * seed + ep - 1).permutation(tr)).to(device)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            nn.functional.l1_loss(net(X[idx]).squeeze(1), target[idx]).backward()
            opt.step()
        val = np.abs(predict(net, X, va_t, scale) - dz[va]).mean()
        if val < best:
            best, best_ep, state = val, ep, copy.deepcopy(net.state_dict())
        if ep % 20 == 0:
            print(f"    seed {seed} epoch {ep}: val MAE {val:.1f}, best {best:.1f} at {best_ep}", flush=True)
        if ep - best_ep >= patience:
            break
    net.load_state_dict(state)
    return net, best_ep, ep, best


def report(z, preds):
    """Tables from test predictions (seeds x images); every figure is the mean over seeds."""
    te = z["test_idx"]
    dz = z["dz_um"][te]
    ok = (np.sign(preds) == np.sign(dz)).mean(0)
    err = np.abs(preds - dz).mean(0)
    size_err = np.abs(np.abs(preds) - np.abs(dz)).mean(0)
    a = np.abs(dz)

    print("\n|dz| bin (µm)    n  sign acc  signed MAE  |dz| MAE")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (a >= lo) & (a < hi)
        print(f"{lo:>4}-{hi:<5} {m.sum():>7} {ok[m].mean():>9.1%} {err[m].mean():>11.1f} {size_err[m].mean():>9.1f}")

    print("\nfrom limit (µm)    n  sign acc  signed MAE  seeds (sign acc)")
    for lim in (0,) + NEAR:
        m = a >= lim
        per_seed = " ".join(f"{(np.sign(p[m]) == np.sign(dz[m])).mean():.1%}" for p in preds)
        print(f">= {lim:<12} {m.sum():>5} {ok[m].mean():>9.1%} {err[m].mean():>11.1f}  {per_seed}")
    for name, side in (("inside (dz<0)", dz < 0), ("outside (dz>0)", dz > 0)):
        m = side & (a >= NEAR[0])
        print(f"{name:<15} >= {NEAR[0]}: n {m.sum()}, sign acc {ok[m].mean():.1%}")

    jitter = np.hypot(z["shift_x_px"], z["shift_y_px"])[te]
    factors = [("seeing FWHM (px)", z["seeing_fwhm_px"][te], [6, 8, 10, 12]),
               ("centre offset (px)", jitter, [0, 1.5, 2.5, 4.3]),
               ("ring SNR", z["ring_snr"][te], [3, 6, 9, 12]),
               ("D/r0", z["d_over_r0"][te], [2, 3, 4, 5])]
    for name, v, edges in factors:
        print(f"\n{name:<20} " + "  ".join(f"n(>={l}) sign acc  MAE" for l in NEAR))
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            inside = (v >= lo) & ((v <= hi) if i == len(edges) - 2 else (v < hi))
            cells = []
            for lim in NEAR:
                m = inside & (a >= lim)
                cells.append(f"{m.sum():>7} {ok[m].mean():>8.1%} {err[m].mean():>5.1f}")
            print(f"{lo}-{hi:<18} " + "  ".join(cells))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else "defocus_stars_v1.npz"
    max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    patience = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(path)
    dz = z["dz_um"].astype(np.float32)
    tr, va, te = z["train_idx"], z["val_idx"], z["test_idx"]
    X = torch.from_numpy(normalise(z["adu"])).to(device)
    scale = float(np.abs(dz).max())
    os.makedirs(OUT, exist_ok=True)
    print(f"device: {device}, max epochs {max_epochs}, patience {patience}, seeds {seeds}")
    preds = []
    for seed in range(seeds):
        t = time.time()
        net, best_ep, last_ep, val = train(X, dz, tr, va, scale, max_epochs, patience, seed, device)
        torch.save({"state_dict": net.state_dict(), "scale_um": scale, "best_epoch": best_ep, "seed": seed},
                   os.path.join(OUT, f"signed_cnn_seed{seed}.pt"))
        p = predict(net, X, torch.from_numpy(te).to(device), scale)
        preds.append(p)
        print(f"  seed {seed}: best epoch {best_ep} of {last_ep}, val MAE {val:.1f}, test signed MAE "
              f"{np.abs(p - dz[te]).mean():.1f}, sign acc {(np.sign(p) == np.sign(dz[te])).mean():.1%} "
              f"({time.time() - t:.0f} s)", flush=True)
    preds = np.stack(preds)
    np.savez(os.path.join(OUT, "signed_test_predictions.npz"), pred_um=preds, test_idx=te)
    report(z, preds)


if __name__ == "__main__":
    main()
