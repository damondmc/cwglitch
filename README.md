# cwglitch

Simulate continuous gravitational-wave signals from glitching pulsars and write them as SFTs, in Gaussian noise
or injected into real data.

## Install

```
uv sync
```

## Waveform model

The signal phase at the solar system barycentre, with $\Delta t = t - t_\mathrm{ref}$, is

$$
\begin{aligned}
\phi(\Delta t) ={}& 2\pi \left[ f_0 \Delta t + \tfrac{1}{2} f_1 \Delta t^2 + \tfrac{1}{6} f_2 \Delta t^3 + \tfrac{1}{24} f_3 \Delta t^4 + \tfrac{1}{120} f_4 \Delta t^5 \right] \\
&+ 2\pi \sum_{\mathrm{glitches}} \Theta(t - t_\mathrm{g}) \left[ (1-Q)\,\Delta\nu\,(t - t_\mathrm{g}) + Q\,\Delta\nu\,\tau \left(1 - e^{-(t - t_\mathrm{g})/\tau}\right) + \tfrac{1}{2}\,\Delta\dot\nu\,(t - t_\mathrm{g})^2 \right]
\end{aligned}
$$

The glitch model: each glitch at
$t_\mathrm{g}$ contributes only for $t > t_\mathrm{g}$ and changes the frequency and spin-down to

$$
f(t) = f_0 + (1-Q)\,\Delta\nu + Q\,\Delta\nu\, e^{-(t - t_\mathrm{g})/\tau}, \qquad
\dot f(t) = f_1 + \Delta\dot\nu - \frac{Q\,\Delta\nu}{\tau}\, e^{-(t - t_\mathrm{g})/\tau}
$$

where

| Symbol | Meaning |
|---|---|
| $f_0, f_1, \dots, f_4$ | GW frequency and spin-downs at $t_\mathrm{ref}$ |
| $t_\mathrm{g}$ | glitch epoch, given as a fraction of the observation span |
| $\Delta\nu = (\Delta\nu/\nu)\, f_0$ | total frequency jump at the glitch |
| $Q$ | healing fraction: $Q\,\Delta\nu$ recovers, $(1-Q)\,\Delta\nu$ is permanent |
| $\tau$ | recovery time scale of the transient part, in seconds |
| $\Delta\dot\nu = (\Delta\dot\nu/\dot\nu)\, f_1$ | permanent spin-down jump |

The strain amplitude is the sum of a constant persistent part $h_0$ and, after each glitch, a
transient-mountain part that is phase-aligned with it (Yim & Jones 2020, eq. 13):

$$
h_0(t) = h_0 + \sum_{\mathrm{glitches}} \Theta(t - t_\mathrm{g})\,
\sqrt{\frac{5}{2}\,\frac{G}{c^3}\,\frac{I}{d^2}\,\frac{Q\,(\Delta\nu/\nu)}{\tau}}\; e^{-(t - t_\mathrm{g})/(2\tau)}
$$

where $I$ is the moment of inertia and $d$ the distance. The transient
amplitude is fixed by the glitch parameters, not by $h_0$, and decays on $2\tau$ because the strain goes as the
square root of the excess spin-down.

Reference: Yim & Jones (2020), *Transient gravitational waves from pulsar post-glitch recoveries*, MNRAS 498, 3138,
[doi:10.1093/mnras/staa2534](https://doi.org/10.1093/mnras/staa2534), [arXiv:2007.05893](https://arxiv.org/abs/2007.05893).

## Example

One signal:

```python
import lalpulsar as lp
import numpy as np

from cwglitch import Glitch, SFTGenerator, Signal, frequency_band, signal_waveform

# One source with one glitch halfway through the data
sig = Signal(
    idx=0,
    f0=100.0,
    f1dot=-1.06e-8,
    phi0=0.0,
    psi=0.785,
    cosi=1.0,
    alpha=6.124,
    delta=1.026,
    glitches=[Glitch(tglitch=0.5, dnu_nu=1e-6, dnu1_nu1=1e-3, healing_factor=0.7, tau=10 * 86400)],
)

# 1000 SFTs of 1800 s in H1, Gaussian noise
tref = 1372426000.0
Tsft = 1800.0
timestamps = tref + Tsft * np.arange(1000)  # SFT start times
Tobs = timestamps[-1] - timestamps[0]

wf, glitches = signal_waveform(
    sig, h0=1e-25, distance_kpc=3.4, moment_of_inertia=1e38, tstart=timestamps[0], Tobs=Tobs, tref=tref
)
fmin, fmax = frequency_band(sig, glitches, timestamps[0], timestamps[-1] + Tsft, tref)
gen = SFTGenerator(sig, wf, "H1", tref, timestamps[0], timestamps[-1] + Tsft, 5.0, Tsft, fmin, fmax, "tukey", 0.001)
sfts = gen.sft_vector(timestamps, sqrtSX=1e-22, rng=np.random.default_rng(0))

spec = lp.SFTFilenameSpec()
spec.path = "out"
spec.window_type = "tukey"
spec.window_param = 0.001
spec.privMisc = "example"
lp.WriteSFTVector2StandardFile(sfts, spec, "example", True)
```

Many signals with randomly drawn parameters, from a config:

```
uv run cwglitch gen-params config/example.yaml   # sample parameters -> <output.dir>/<output.params_file>
uv run cwglitch simulate   config/example.yaml   # write SFTs       -> <output.dir>/<output.sft_label>/
uv run cwglitch run        config/example.yaml   # both
```
