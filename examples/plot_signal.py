"""Generate one glitching signal in H1 and plot its amplitude, intrinsic frequency and SFT power."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cwglitch import Glitch, SFTGenerator, Signal, frequency_band, signal_waveform

DIR = Path(__file__).resolve().parent
plt.style.use(DIR / "cwglitch.mplstyle")

sig = Signal(
    idx=0,
    f0=100.0,
    f1dot=-1.06e-8,
    phi0=0.0,
    psi=0.785,
    cosi=1.0,
    alpha=6.124,
    delta=1.026,
    glitches=[Glitch(tglitch=0.5, dnu_nu=1e-6, dnu1_nu1=1e-3, healing_factor=0.7, tau=86400.0)],
)
h0, distance_kpc, moment_of_inertia = 1e-25, 3.4, 1e38

tref = 1372426000.0
Tsft = 1800.0
timestamps = tref + Tsft * np.arange(480)
Tobs = timestamps[-1] - timestamps[0]

wf, glitches = signal_waveform(sig, h0, distance_kpc, moment_of_inertia, timestamps[0], Tobs, tref)
fmin, fmax = frequency_band(sig, glitches, timestamps[0], timestamps[-1] + Tsft, tref)
generator = SFTGenerator(
    sig, wf, "H1", tref, timestamps[0], timestamps[-1] + Tsft, 5.0, Tsft, fmin, fmax, "tukey", 0.001
)
sfts = generator.sft_vector(timestamps, sqrtSX=1e-24, rng=np.random.default_rng(0))

n_days = (timestamps - timestamps[0]) / 86400.0
t_glitch_days = (glitches[0].t + tref - timestamps[0]) / 86400.0
dt = timestamps - tref
_, aplus, _ = wf(dt)
amplitude = aplus / ((1.0 + sig.cosi**2) / 2.0)


def mean_frequency(waveform, t, half_window=60.0):
    """Mean frequency over [t - half_window, t + half_window]: the phase difference across the window over 2 pi."""
    dphi = waveform(t + half_window)[0] - waveform(t - half_window)[0]
    return dphi / (2.0 * half_window) / (2.0 * math.pi)


# Same source with no glitch, to isolate what the glitch adds
no_glitch = sig.model_copy(update={"glitches": []})
wf_no_glitch, _ = signal_waveform(no_glitch, h0, distance_kpc, moment_of_inertia, timestamps[0], Tobs, tref)
freq = mean_frequency(wf, dt)
freq_glitch = freq - mean_frequency(wf_no_glitch, dt)

# SFT power around the signal
power = np.array([np.abs(sfts.data[k].data.data) for k in range(sfts.length)])
f_axis = fmin + np.arange(power.shape[1]) / Tsft
sel = (f_axis > sig.f0 - 0.03) & (f_axis < sig.f0 + 0.03)

fig = plt.figure(figsize=(9, 9))
grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])
ax_strain = fig.add_subplot(grid[0, :])
ax_freq = fig.add_subplot(grid[1, 0], sharex=ax_strain)
ax_freq_glitch = fig.add_subplot(grid[1, 1], sharex=ax_strain)
ax_spectrogram = fig.add_subplot(grid[2, :], sharex=ax_strain)

ax_strain.plot(n_days, amplitude / 1e-25, label=r"$h_0(t)$")
ax_strain.set_ylabel(r"$h_0(t)\ [10^{-25}]$")
ax_strain.set_title(r"Strain: persistent $h_0$ plus transient mountain")

ax_freq.plot(n_days, freq, label=r"$f(t)$")
ax_freq.set_ylabel(r"$f(t)\ [\mathrm{Hz}]$")
ax_freq.set_title(r"Intrinsic frequency")
ax_freq.ticklabel_format(axis="y", useOffset=False)

ax_freq_glitch.plot(n_days, freq_glitch * 1e6, label=r"$f_{\mathrm{glitch}}(t)$")
ax_freq_glitch.set_ylabel(r"$f_{\mathrm{glitch}}(t)\ [\mu\mathrm{Hz}]$")
ax_freq_glitch.set_title(r"Glitch part: $(1-Q)\Delta\nu$ step, $Q\Delta\nu$ recovery, $\Delta\dot\nu$ step")

im = ax_spectrogram.pcolormesh(n_days, f_axis[sel], power[:, sel].T, shading="nearest", cmap="viridis")
ax_spectrogram.set_ylabel(r"$f\ [\mathrm{Hz}]$")
ax_spectrogram.ticklabel_format(axis="y", useOffset=False)
ax_spectrogram.set_title(r"H1 SFT amplitude $|\tilde{h}(f)|$")
fig.colorbar(im, cax=ax_spectrogram.inset_axes((1.02, 0.0, 0.02, 1.0)), label=r"$|\tilde{h}|$")

for ax in (ax_strain, ax_freq, ax_freq_glitch, ax_spectrogram):
    ax.axvline(t_glitch_days, color="k", ls="--", lw=0.8, label=r"$t_{\mathrm{g}}$")
    ax.tick_params(labelbottom=True)
    ax.set_xlim(n_days[0], n_days[-1])
    ax.set_xlabel(r"$t - t_{\mathrm{start}}\ [\mathrm{days}]$")
for ax in (ax_strain, ax_freq, ax_freq_glitch):
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

fig.tight_layout()
out = DIR / "signal.png"
fig.savefig(out)
print(f"wrote {out}")
