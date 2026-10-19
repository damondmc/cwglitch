"""Phase and amplitude model for a continuous wave with glitches.

Glitch model after Yim & Jones (2020), "Transient gravitational waves from pulsar
post-glitch recoveries", MNRAS 498, 3138, doi:10.1093/mnras/staa2534, arXiv:2007.05893

With dt = t - tref, the pre-glitch phase is

    phi(dt) = 2 pi [ f0 dt + f1 dt^2/2 + f2 dt^3/6 + f3 dt^4/24 + f4 dt^5/120 ]

Each glitch at epoch tg adds, for d = dt - tg > 0,

    phi += 2 pi [ df_p d + df_t tau (1 - exp(-d/tau)) + df1_p d^2/2 ]

so the instantaneous frequency and spindown become

    f(dt)  = f0 + df_p + df_t exp(-d/tau)
    f1(dt) = f1 + df1_p - (df_t/tau) exp(-d/tau)

The amplitude follows h0(t) = h0 sqrt(|f1(dt)| / f(dt)) / sqrt(|f1| / f0).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cwglitch.params import Glitch, Signal

Waveform = Callable[[float], tuple[float, float, float]]


@dataclass(frozen=True)
class PhysicalGlitch:
    t: float  # glitch epoch relative to tref, seconds
    df_p: float  # permanent frequency jump, Hz
    df_t: float  # transient frequency jump, Hz
    df1_p: float  # permanent spindown jump, Hz/s
    tau: float  # seconds


def physical_glitch(glitch: Glitch, f0: float, f1: float, tstart: float, Tobs: float, tref: float) -> PhysicalGlitch:
    """Convert relative glitch parameters to Hz, Hz/s and seconds after tref."""
    dnu = glitch.dnu_nu * f0
    return PhysicalGlitch(
        t=tstart + glitch.tglitch * Tobs - tref,
        df_p=(1.0 - glitch.Q) * dnu,
        df_t=glitch.Q * dnu,
        df1_p=glitch.dnu1_nu1 * f1,
        tau=glitch.tau,
    )


def make_waveform(sig: Signal, h0: float, glitches: Sequence[PhysicalGlitch]) -> Waveform:
    """Return wf(dt) -> (phase, a_plus, a_cross) as CWSimulator expects."""
    if sig.f1 == 0:
        raise ValueError("f1 is zero.")
    h0_scale = h0 / math.sqrt(abs(sig.f1) / sig.f0)

    def wf(dt: float) -> tuple[float, float, float]:
        dphi = (
            sig.f0 * dt + sig.f1 * dt**2 / 2.0 + sig.f2 * dt**3 / 6.0 + sig.f3 * dt**4 / 24.0 + sig.f4 * dt**5 / 120.0
        )
        f_eff = sig.f0
        f1_eff = sig.f1
        for glitch in glitches:
            if dt > glitch.t:
                d = dt - glitch.t
                decay = math.exp(-d / glitch.tau)
                dphi += glitch.df_p * d + glitch.df_t * glitch.tau * (1.0 - decay) + glitch.df1_p * d**2 / 2.0
                f_eff += glitch.df_p + glitch.df_t * decay
                f1_eff += glitch.df1_p - glitch.df_t / glitch.tau * decay
        if glitches:
            if f1_eff == 0:
                raise ValueError("Effective f1 is zero.")
            h0_t = h0_scale * math.sqrt(abs(f1_eff) / f_eff)
        else:
            h0_t = h0
        ap = h0_t * (1.0 + sig.cosi**2) / 2.0
        ax = h0_t * sig.cosi
        return 2.0 * math.pi * dphi, ap, ax

    return wf


def signal_waveform(
    sig: Signal, h0: float, tstart: float, Tobs: float, tref: float
) -> tuple[Waveform, list[PhysicalGlitch]]:
    """Build the waveform for a sampled signal. Also returns the physical glitches, needed for the SFT band."""
    glitches = [physical_glitch(glitch, sig.f0, sig.f1, tstart, Tobs, tref) for glitch in sig.glitches]
    return make_waveform(sig, h0, glitches), glitches
