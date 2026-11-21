"""Rebuild the toy defocused-star benchmark set (v0), seed 0. Not physical optics: see README."""
import numpy as np, torch

def gen(n, s=64):
    d = torch.empty(n).uniform_(-1, 1)
    y, x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
    r = ((x - s/2)**2 + (y - s/2)**2).sqrt()[None]
    R = 2 + 12*d.abs()[:, None, None]
    w = 1.5 + 0.5*d[:, None, None]
    img = torch.exp(-((r - R)**2) / (2*w**2))
    counts = torch.poisson(img*200 + 10)          # integer photon counts
    return counts[:, None], d

if __name__ == "__main__":
    torch.manual_seed(0)
    C, Y = gen(20000)
    np.savez_compressed("toy_defocus_stars_v0.npz",
        counts=C.numpy().astype(np.uint16), d=Y.numpy().astype(np.float32),
        train_idx=np.arange(0, 18000), test_idx=np.arange(18000, 20000), scale=np.float32(200.0))
