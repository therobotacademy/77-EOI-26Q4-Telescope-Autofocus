"""Classical HFR baseline on dataset v1: estimate blur size |dz| (µm) from one 64x64 star.

HFR (half-flux radius) is measured per image, then mapped to |dz| with a monotone curve fitted on the
train split (median |dz| per HFR quantile). The measurement variant is chosen on the val split; the test
split is only used for the final report. HFR carries no sign, so only |dz| is estimated.
Needs numpy only.  Usage: python hfr_baseline.py [defocus_stars_v1.npz]"""
import json, sys
import numpy as np

BORDER = 4                      # px used for sky level and noise (same as train_check_v1.py)
EDGES = [0, 200, 400, 800, 1200, 1600]


def box3(x):
    """3x3 mean filter on a stack of images, used only to find the star."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return sum(p[:, i:i + x.shape[1], j:j + x.shape[2]] for i in range(3) for j in range(3)) / 9


def hfr(adu, k_sigma, aperture, chunk=1000):
    """Radius (px) enclosing half of the background-subtracted flux inside `aperture` px of the centroid.
    k_sigma sets the detection threshold (in border noise units) of the smoothed image for the centroid."""
    h, w = adu.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty(len(adu), np.float32)
    for s in range(0, len(adu), chunk):
        a = adu[s:s + chunk].astype(np.float32)
        n = len(a)
        border = np.concatenate([a[:, :BORDER].reshape(n, -1), a[:, -BORDER:].reshape(n, -1),
                                 a[:, BORDER:-BORDER, :BORDER].reshape(n, -1),
                                 a[:, BORDER:-BORDER, -BORDER:].reshape(n, -1)], 1)
        a = a - np.median(border, 1)[:, None, None]
        sm = box3(a)
        # centroid of the pixels clearly above the noise (noise of a 3x3 mean = std / 3)
        thr = k_sigma * border.std(1)[:, None, None] / 3
        wgt = np.where(sm > thr, sm, 0)
        tot = wgt.sum((1, 2)) + 1e-6
        cx = (wgt * xx).sum((1, 2)) / tot
        cy = (wgt * yy).sum((1, 2)) / tot
        r = np.hypot(xx - cx[:, None, None], yy - cy[:, None, None]).reshape(n, -1)
        flux = np.where(r <= aperture, a.reshape(n, -1), 0)   # no clipping: noise averages to zero
        order = np.argsort(r, 1)
        rs = np.take_along_axis(r, order, 1)
        cum = np.cumsum(np.take_along_axis(flux, order, 1), 1)
        total = np.take_along_axis(flux, order, 1).sum(1)
        first = np.argmax(cum >= 0.5 * total[:, None], 1)
        out[s:s + n] = rs[np.arange(n), first]
    return out


def fit_curve(h, target, bins=80):
    """Monotone HFR -> |dz| curve: median target in equal-count HFR bins, then a running maximum."""
    groups = np.array_split(np.argsort(h), bins)
    xs = np.array([np.median(h[g]) for g in groups])
    ys = np.maximum.accumulate([np.median(target[g]) for g in groups])
    return xs, ys


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else "defocus_stars_v1.npz"
    z = np.load(path)
    adu, dz = z["adu"], np.abs(z["dz_um"])
    tr, va, te = z["train_idx"], z["val_idx"], z["test_idx"]
    tel = json.loads(str(z["config"]))["telescope"]
    # depth of focus: quarter-wave (Rayleigh) tolerance, ±2 λ N²
    dof = 2 * tel["wavelength_um"] * (tel["focal_mm"] / tel["aperture_mm"]) ** 2

    best = None
    print("variant selection on val split (MAE of |dz|, µm)")
    for k_sigma in (1.0, 2.0, 3.0):
        for aperture in (26, 30, 34, 38, 42, 46):
            h = hfr(adu, k_sigma, aperture)
            xs, ys = fit_curve(h[tr], dz[tr])
            mae = np.abs(np.interp(h[va], xs, ys) - dz[va]).mean()
            print(f"  k_sigma={k_sigma:.0f} aperture={aperture} px: {mae:6.1f}", flush=True)
            if best is None or mae < best[0]:
                best = (mae, k_sigma, aperture, h, xs, ys)
    _, k_sigma, aperture, h, xs, ys = best

    y = dz[te]
    err = np.abs(np.interp(h[te], xs, ys) - y)
    const = np.abs(np.median(dz[tr]) - y)
    print(f"\nchosen: k_sigma={k_sigma:.0f}, aperture={aperture} px")
    print(f"test MAE of |dz|: {err.mean():.1f} µm = {err.mean() / dof:.1f} depths of focus (±{dof:.1f} µm); "
          f"median error {np.median(err):.1f} µm; constant guess {const.mean():.1f} µm")
    print("\n|dz| bin (µm)   n    MAE (µm)  MAE (DoF)  median err (µm)  bias (µm)")
    signed = np.interp(h[te], xs, ys) - y
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (y >= a) & (y < b)
        if m.any():
            print(f"{a:>5}-{b:<5} {m.sum():>5} {err[m].mean():>9.0f} {err[m].mean() / dof:>10.1f} "
                  f"{np.median(err[m]):>16.0f} {signed[m].mean():>10.0f}")


if __name__ == "__main__":
    main()
