"""Generate defocused-star dataset v1 with star_sim.py, in shards saved as they finish, then merge.

Usage:  python make_dataset_v1.py [n_images] [out.npz]
Shard k uses seed 1000 + k, so any shard can be regenerated alone.
"""
import sys, os, time, json
from dataclasses import asdict
import numpy as np
from star_sim import StarSimulator, Telescope, Ranges

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
out = sys.argv[2] if len(sys.argv) > 2 else "defocus_stars_v1.npz"
shard = 2000
tel, rng = Telescope(), Ranges()
os.makedirs("shards", exist_ok=True)
t = time.time()
for k in range((n + shard - 1) // shard):
    path = f"shards/shard_{k:02d}.npz"
    if os.path.exists(path):
        continue
    sim = StarSimulator(tel, rng, seed=1000 + k)
    X, meta = sim.generate(min(shard, n - k * shard))
    np.savez(path + ".tmp.npz", adu=X, **meta)
    os.replace(path + ".tmp.npz", path)
    print(f"shard {k} done, {time.time() - t:.0f} s", flush=True)

parts = [np.load(f"shards/shard_{k:02d}.npz") for k in range((n + shard - 1) // shard)]
keys = [k for k in parts[0].files]
data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
n_tr, n_va = int(0.8 * n), int(0.1 * n)
np.savez_compressed(out, **data,
                    train_idx=np.arange(0, n_tr), val_idx=np.arange(n_tr, n_tr + n_va),
                    test_idx=np.arange(n_tr + n_va, n),
                    config=json.dumps({"telescope": asdict(tel), "ranges": asdict(rng),
                                       "seeds": "1000 + shard index, 2000 images per shard",
                                       "pupil_px": 256, "grid": 512}))
print(f"merged {n} images -> {out}, total {time.time() - t:.0f} s", flush=True)
