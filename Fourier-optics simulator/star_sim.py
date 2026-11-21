"""Physical simulator of defocused star images for single-image telescope autofocus (dataset v1).

Each image is a 64x64 cut-out in ADU, labelled with the signed focus error dz (µm at the focal plane,
dz > 0 = sensor outside focus). The star is computed with Fourier optics:

  pupil (aperture + central obstruction [+ spider vanes])
  x exp(i·2π·W), with W = defocus(dz) + spherical aberration + random low-order aberrations + turbulence
  -> |FFT|² = PSF -> resampled to the camera pixel scale -> seeing blur (Moffat) -> flux + sky
  -> Poisson noise, read noise, gain, bias, rounding, saturation.

The spherical aberration term is what makes the sign learnable: with it, the same |dz| gives a
different light distribution inside and outside focus, as in a real telescope.

All defaults are TYPICAL values (200 mm f/5 Newtonian, 3.76 µm pixels), to be replaced by the
student's real telescope and camera. Change them in Telescope / Ranges below.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Telescope:
    aperture_mm: float = 200.0
    focal_mm: float = 1000.0          # f/5
    obstruction: float = 0.30         # secondary diameter / aperture
    spider_vanes: int = 0             # 0, 3 or 4 vanes holding the secondary
    vane_width: float = 0.01          # fraction of the aperture
    wavelength_um: float = 0.55
    pixel_um: float = 3.76            # camera pixel size
    gain_e_per_adu: float = 1.0
    bias_adu: float = 100.0
    saturation_adu: int = 65535
    cutout: int = 64                  # cut-out side in camera pixels
    spherical_waves: float = 0.4      # fixed spherical aberration of the optics (Z11, waves RMS-like coefficient)

    @property
    def fnum(self):
        return self.focal_mm / self.aperture_mm

    def dz_per_wave(self):
        """Focal-plane shift (µm) that gives 1 wave peak-to-valley of defocus: dz = 8 N² λ W."""
        return 8 * self.fnum ** 2 * self.wavelength_um


@dataclass
class Ranges:
    """Per-image random parameters (uniform in [lo, hi] unless noted)."""
    defocus_waves: tuple = (-11.0, 11.0)      # P-V defocus; sets dz. ±11 waves = ±1210 µm; donut radius ≈ 16 px + seeing, fits 64 px
    seeing_fwhm_px: tuple = (6.0, 12.0)       # Moffat FWHM in camera pixels (real frame fit: 10)
    moffat_beta: tuple = (2.5, 4.5)
    d_over_r0: tuple = (2.0, 5.0)             # turbulence strength across the aperture (short exposure)
    low_order_waves: float = 0.15             # RMS of random astigmatism / coma per image
    center_jitter_px: float = 3.0             # max shift of the star from the cut-out centre
    sky_e: tuple = (50.0, 800.0)              # sky background, electrons per pixel
    read_noise_e: tuple = (3.0, 10.0)
    ring_snr: tuple = (3.0, 12.0)             # mean donut signal / background noise (real frame: about 6.5)


class StarSimulator:
    def __init__(self, tel: Telescope = Telescope(), rng: Ranges = Ranges(), pupil_px=256, grid=512, seed=0):
        self.tel, self.rng = tel, rng
        self.Dp, self.N = pupil_px, grid
        self.gen = torch.Generator().manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        # FFT pixel = (Dp/N)·λ/D ; camera pixel = pixel/(λ·N#) in λ/D units
        self.fft_px_lod = pupil_px / grid
        self.cam_px_lod = tel.pixel_um / (tel.wavelength_um * tel.fnum)
        self.zoom = self.cam_px_lod / self.fft_px_lod            # FFT pixels per camera pixel
        n = grid
        c = (torch.arange(n) - n / 2) / (pupil_px / 2)              # pupil coords, 1 = edge
        y, x = torch.meshgrid(c, c, indexing="ij")
        self.x, self.y = x, y
        r = (x ** 2 + y ** 2).sqrt()
        self.rho, self.theta = r, torch.atan2(y, x)
        amp = ((r <= 1) & (r >= tel.obstruction)).float()
        for k in range(tel.spider_vanes):          # vanes = half-lines from the centre, evenly spaced
            a = 2 * math.pi * k / tel.spider_vanes
            along = x * math.cos(a) + y * math.sin(a)
            across = (x * math.sin(a) - y * math.cos(a)).abs()
            amp = amp * ~((along > 0) & (across < tel.vane_width / 2 * 2))  # width as fraction of the radius·2
        amp = amp.float()
        self.amp = amp
        r2 = r ** 2
        self.Z = {  # unnormalised Zernike-like shapes on the unit pupil
            "defocus": r2,                                  # P-V = 1 wave at the edge
            "spherical": 6 * r2 ** 2 - 6 * r2 + 1,
            "astig0": r2 * torch.cos(2 * self.theta),
            "astig45": r2 * torch.sin(2 * self.theta),
            "comax": (3 * r2 - 2) * r * torch.cos(self.theta),
            "comay": (3 * r2 - 2) * r * torch.sin(self.theta),
        }
        # Kolmogorov filter for phase screens (in pupil-grid frequency units)
        f = torch.fft.fftfreq(n)
        fy, fx = torch.meshgrid(f, f, indexing="ij")
        fr = (fx ** 2 + fy ** 2).sqrt()
        fr[0, 0] = 1.0
        filt = fr ** (-11 / 6)
        filt[0, 0] = 0.0
        self.kolmo = filt

    def _u(self, lo, hi, n):
        return lo + (hi - lo) * torch.rand(n, generator=self.gen)

    def _phase_screens(self, d_over_r0):
        """Kolmogorov screens scaled so the phase structure across the pupil ~ (D/r0)^(5/3) (waves, approx)."""
        n = d_over_r0.shape[0]
        noise = torch.randn(n, self.N, self.N, generator=self.gen, dtype=torch.float32)
        scr = torch.fft.ifft2(torch.fft.fft2(noise) * self.kolmo).real
        inside = self.amp > 0
        scr = scr - scr[:, inside].mean(1)[:, None, None]
        rms = scr[:, inside].std(1)[:, None, None]
        target_rms_waves = 1.03 ** 0.5 * d_over_r0[:, None, None] ** (5 / 6) / (2 * math.pi)  # Noll 1976, tilt included
        return scr / rms * target_rms_waves

    def _moffat_kernel(self, fwhm, beta, size=47):
        c = torch.arange(size) - size // 2
        y, x = torch.meshgrid(c, c, indexing="ij")
        r2 = (x ** 2 + y ** 2)[None].float()
        alpha = fwhm / (2 * torch.sqrt(2 ** (1 / beta) - 1))
        k = (1 + r2 / alpha[:, None, None] ** 2) ** (-beta[:, None, None])
        return k / k.sum((1, 2), keepdim=True)

    @torch.no_grad()
    def batch(self, n, fixed=None, clean=False):
        """fixed: optional dict of tensors/floats overriding the random draws
        (defocus_waves, seeing_fwhm_px, moffat_beta, d_over_r0, low_order, shift).
        clean=True returns the noise-free unit-flux image (for fitting and inspection)."""
        t, g, R = self.tel, self.gen, self.rng
        fx = fixed or {}
        full = lambda v: torch.as_tensor(v, dtype=torch.float32).expand(n).clone()
        W_def = full(fx["defocus_waves"]) if "defocus_waves" in fx else self._u(*R.defocus_waves, n)
        dz = W_def * t.dz_per_wave()
        lo = fx.get("low_order", R.low_order_waves)
        coef = {k: torch.randn(n, generator=g) * lo for k in ("astig0", "astig45", "comax", "comay")}
        d_r0 = full(fx["d_over_r0"]) if "d_over_r0" in fx else self._u(*R.d_over_r0, n)
        # sub-pixel centre shift, applied as a pupil tilt (exact Fourier shift)
        sh = (torch.rand(n, 2, generator=g) * 2 - 1) * fx.get("shift", R.center_jitter_px)
        sh_fft = sh * self.zoom
        W = (W_def[:, None, None] * self.Z["defocus"] + t.spherical_waves * self.Z["spherical"])
        for k, v in coef.items():
            W = W + v[:, None, None] * self.Z[k]
        W = W + self._phase_screens(d_r0)
        u = torch.arange(self.N) - self.N / 2
        tilt = (sh_fft[:, 0, None, None] * u[None, None, :] + sh_fft[:, 1, None, None] * u[None, :, None]) / self.N
        field = self.amp * torch.exp(2j * math.pi * (W + tilt))
        psf = torch.fft.fftshift(torch.fft.fft2(field).abs() ** 2, dim=(-2, -1))
        # crop the camera field and integrate into camera pixels
        side = int(math.ceil(t.cutout * self.zoom))
        c0 = self.N // 2 - side // 2
        psf = psf[:, c0:c0 + side, c0:c0 + side]
        img = F.adaptive_avg_pool2d(psf[:, None], t.cutout)[:, 0]
        img = img / img.sum((1, 2), keepdim=True)                      # unit flux in the cut-out
        # seeing
        fwhm = full(fx["seeing_fwhm_px"]) if "seeing_fwhm_px" in fx else self._u(*R.seeing_fwhm_px, n)
        beta = full(fx["moffat_beta"]) if "moffat_beta" in fx else self._u(*R.moffat_beta, n)
        ker = self._moffat_kernel(fwhm, beta)
        img = F.conv2d(img[None], ker[:, None], padding=ker.shape[-1] // 2, groups=n)[0]
        img = img / img.sum((1, 2), keepdim=True)
        if clean:
            return img
        # sensor: choose flux so the mean signal on the donut is ring_snr × background noise
        sky = self._u(*R.sky_e, n)
        rn = self._u(*R.read_noise_e, n)
        snr = self._u(*R.ring_snr, n)
        sigma_bg = (sky + rn ** 2).sqrt()
        thr = img.amax((1, 2), keepdim=True) * 0.25
        ring_mean = (img * (img > thr)).sum((1, 2)) / (img > thr).sum((1, 2))
        flux = snr * sigma_bg / ring_mean
        e = torch.poisson(img * flux[:, None, None] + sky[:, None, None], generator=g)
        e = e + torch.randn(e.shape, generator=g) * rn[:, None, None]
        adu = (e / t.gain_e_per_adu + t.bias_adu).round().clamp(0, t.saturation_adu)
        meta = dict(dz_um=dz, defocus_waves=W_def, seeing_fwhm_px=fwhm, moffat_beta=beta, d_over_r0=d_r0,
                    shift_x_px=sh[:, 0], shift_y_px=sh[:, 1], sky_e=sky, read_noise_e=rn, ring_snr=snr,
                    flux_e=flux, **{f"{k}_waves": v for k, v in coef.items()})
        return adu.to(torch.int32), {k: v.float() for k, v in meta.items()}

    def generate(self, n, batch=32):
        imgs, metas = [], []
        for i in range(0, n, batch):
            a, m = self.batch(min(batch, n - i))
            imgs.append(a); metas.append(m)
        X = torch.cat(imgs).numpy().astype(np.uint16)
        meta = {k: torch.cat([m[k] for m in metas]).numpy() for k in metas[0]}
        return X, meta


def preprocess(adu, bias=100.0):
    """PLACEHOLDER for the student's 'preprocesada' step (unknown): subtract the background (median),
    divide by a robust noise estimate, clip to [-1, p99.5] and scale to [0, 1]. Replace with the real one,
    and apply the same function to synthetic and real images."""
    x = adu.astype(np.float32) - bias
    med = np.median(x, axis=(-2, -1), keepdims=True)
    mad = np.median(np.abs(x - med), axis=(-2, -1), keepdims=True) * 1.4826 + 1e-6
    z = (x - med) / mad
    hi = np.percentile(z, 99.5, axis=(-2, -1), keepdims=True)
    return np.clip((z + 1) / (hi + 1), 0, 1)
