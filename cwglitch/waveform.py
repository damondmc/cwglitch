"""Phase and amplitude model for a continuous wave with glitches.

Glitch model after Yim & Jones (2020), "Transient gravitational waves from pulsar
post-glitch recoveries", MNRAS 498, 3138, doi:10.1093/mnras/staa2534, arXiv:2007.05893

With dt = t - tref, the pre-glitch phase is

    phi(dt) = 2 pi [ f0 dt + f1dot dt^2/2 + f2dot dt^3/6 + f3dot dt^4/24 + f4dot dt^5/120 ]

Each glitch at epoch tg adds, for d = dt - tg > 0 (their eq. 1 integrated, with f = 2 nu),

    phi += 2 pi [ df_permanent d + df_transient tau (1 - exp(-d/tau)) + df1dot_permanent d^2/2 ]

The persistent amplitude h0 is constant. Each glitch adds the transient-mountain strain:

    h0_transient(d) = sqrt( (5/2) (G/c^3) (I/d_src^2) healing_factor (dnu/nu) / tau ) exp(-d / (2 tau))
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import lal
import numpy as np

from cwglitch.params import Glitch, Signal

Waveform = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class PhysicalGlitch:
    t: float  # glitch epoch relative to tref, seconds
    df_permanent: float  # permanent frequency jump, Hz
    df_transient: float  # transient frequency jump, Hz
    df1dot_permanent: float  # permanent spindown jump, Hz/s
    tau: float  # seconds
    h0_transient: float  # peak transient-mountain strain, Yim & Jones (2020) eq. 13 at d = 0


def physical_glitch(
    glitch: Glitch,
    f0: float,
    f1dot: float,
    distance_kpc: float,
    moment_of_inertia: float,
    tstart: float,
    Tobs: float,
    tref: float,
) -> PhysicalGlitch:
    """Convert relative glitch parameters to Hz, Hz/s, seconds after tref and strain."""
    dnu = glitch.dnu_nu * f0
    dist = distance_kpc * 1e3 * lal.PC_SI
    h0_transient = math.sqrt(
        2.5 * lal.G_SI / lal.C_SI**3 * moment_of_inertia / dist**2 * glitch.healing_factor * glitch.dnu_nu / glitch.tau
    )
    return PhysicalGlitch(
        t=tstart + glitch.tglitch * Tobs - tref,
        df_permanent=(1.0 - glitch.healing_factor) * dnu,
        df_transient=glitch.healing_factor * dnu,
        df1dot_permanent=glitch.dnu1_nu1 * f1dot,
        tau=glitch.tau,
        h0_transient=h0_transient,
    )


def make_waveform(sig: Signal, h0: float, glitches: Sequence[PhysicalGlitch]) -> Waveform:
    """Return wf(dt) -> (phase, a_plus, a_cross), vectorised over an array of dt."""

    def wf(dt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """dt must be sorted ascending; glitch terms are applied to the slice after each epoch."""
        dt = np.atleast_1d(np.asarray(dt, dtype=float))
        dphi = (
            sig.f0 * dt
            + sig.f1dot * dt**2 / 2.0
            + sig.f2dot * dt**3 / 6.0
            + sig.f3dot * dt**4 / 24.0
            + sig.f4dot * dt**5 / 120.0
        )
        amplitude = np.full_like(dt, h0)
        for glitch in glitches:
            k = int(np.searchsorted(dt, glitch.t, side="right"))
            d = dt[k:] - glitch.t
            decay = np.exp(-d / glitch.tau)
            dphi[k:] += (
                glitch.df_permanent * d
                + glitch.df_transient * glitch.tau * (1.0 - decay)
                + glitch.df1dot_permanent * d**2 / 2.0
            )
            amplitude[k:] += glitch.h0_transient * np.sqrt(decay)
        ap = amplitude * (1.0 + sig.cosi**2) / 2.0
        ax = amplitude * sig.cosi
        return 2.0 * math.pi * dphi, ap, ax

    return wf


def signal_waveform(
    sig: Signal,
    h0: float,
    distance_kpc: float,
    moment_of_inertia: float,
    tstart: float,
    Tobs: float,
    tref: float,
) -> tuple[Waveform, list[PhysicalGlitch]]:
    """Build the waveform for a sampled signal. Also returns the physical glitches, needed for the SFT band."""
    glitches = [
        physical_glitch(glitch, sig.f0, sig.f1dot, distance_kpc, moment_of_inertia, tstart, Tobs, tref)
        for glitch in sig.glitches
    ]
    return make_waveform(sig, h0, glitches), glitches
