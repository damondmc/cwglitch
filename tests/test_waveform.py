import math

import numpy as np
import pytest

from cwglitch import Glitch, Signal, physical_glitch, signal_waveform
from cwglitch.waveform import Waveform

from .conftest import DISTANCE_KPC, H0, MOMENT_OF_INERTIA

TOBS = 4e6


def mean_frequency(waveform: Waveform, t: float, half_window: float = 50.0) -> float:
    """Phase difference over [t - half_window, t + half_window] over 2 pi.

    The phase reaches 1e10 rad, so a wide window keeps float round-off below 1e-8 Hz.
    """
    phase = waveform(np.array([t - half_window, t + half_window]))[0]
    return float(phase[1] - phase[0]) / (2 * half_window) / (2 * math.pi)


def test_frequency_jumps_by_dnu_at_glitch(signal: Signal) -> None:
    waveform, glitches = signal_waveform(signal, H0, DISTANCE_KPC, MOMENT_OF_INERTIA, 0.0, TOBS, 0.0)
    glitch = glitches[0]
    assert glitch.t == pytest.approx(0.5 * TOBS)
    dnu = signal.glitches[0].dnu_nu * signal.f0
    jump = mean_frequency(waveform, glitch.t + 60.0) - mean_frequency(waveform, glitch.t - 60.0)
    jump -= signal.f1dot * 120.0
    assert jump == pytest.approx(dnu, rel=1e-3)


def test_only_permanent_part_remains_long_after(signal: Signal) -> None:
    waveform, glitches = signal_waveform(signal, H0, DISTANCE_KPC, MOMENT_OF_INERTIA, 0.0, TOBS, 0.0)
    glitch = glitches[0]
    t = glitch.t + 50 * glitch.tau
    secular = signal.f0 + signal.f1dot * t
    residual = mean_frequency(waveform, t) - secular
    expected = glitch.df_permanent + glitch.df1dot_permanent * (t - glitch.t)
    assert residual == pytest.approx(expected, rel=1e-4)


def test_amplitude_follows_yim_jones(signal: Signal) -> None:
    waveform, glitches = signal_waveform(signal, H0, DISTANCE_KPC, MOMENT_OF_INERTIA, 0.0, TOBS, 0.0)
    glitch = glitches[0]
    _, aplus, _ = waveform(np.array([glitch.t - 1.0, glitch.t + 1e-3, glitch.t + 2 * glitch.tau]))
    amplitude = aplus / ((1 + signal.cosi**2) / 2)
    # abs=0 makes approx purely relative; its default abs=1e-12 would swamp any rel tolerance on 1e-25 strains
    assert amplitude[0] == pytest.approx(H0, rel=1e-6, abs=0)
    assert amplitude[1] == pytest.approx(H0 + glitch.h0_transient, rel=1e-6, abs=0)
    assert amplitude[2] == pytest.approx(H0 + glitch.h0_transient / math.e, rel=1e-6, abs=0)


def test_transient_strain_matches_yim_jones_table3() -> None:
    # Vela, MJD 57734 glitch: dnu/nu = 1.431e-6, Q = 0.0085, d = 0.28 kpc, I = 1e38 -> h0 = 9.2e-25.
    # Their eq. 14 uses dnudot/nudot = 7.335e-2 with nudot0/nu0 = -1.567e-11/11.195; pick tau to match.
    nu0, nudot0 = 11.195, -1.567e-11
    healing_factor, dnu_nu = 0.0085, 1.431e-6
    tau = healing_factor * dnu_nu / (-nudot0 / nu0 * 7.335e-2)
    glitch = Glitch(tglitch=0.5, dnu_nu=dnu_nu, dnu1_nu1=0.0, healing_factor=healing_factor, tau=tau)
    physical = physical_glitch(glitch, 2 * nu0, 2 * nudot0, 0.28, 1e38, 0.0, 1.0, 0.0)
    assert physical.h0_transient == pytest.approx(9.2e-25, rel=0.01, abs=0)


def test_multiple_glitches_apply_only_after_their_epochs() -> None:
    glitch_params = [
        Glitch(tglitch=x, dnu_nu=1e-6, dnu1_nu1=1e-3, healing_factor=0.5, tau=1e5) for x in (0.2, 0.5, 0.9)
    ]
    signal = Signal(
        idx=0,
        f0=100.0,
        f1dot=-1e-8,
        f2dot=1e-20,
        phi0=0.0,
        psi=0.0,
        cosi=1.0,
        alpha=0.0,
        delta=0.0,
        glitches=glitch_params,
    )
    waveform, glitches = signal_waveform(signal, H0, DISTANCE_KPC, MOMENT_OF_INERTIA, -1e6, TOBS, 0.0)
    dt = np.sort(np.random.default_rng(1).uniform(-1e6, 3e6, 20000))  # starts well before the first glitch
    phase = waveform(dt)[0]
    expected = 2 * math.pi * (signal.f0 * dt + signal.f1dot * dt**2 / 2 + signal.f2dot * dt**3 / 6)
    for glitch in glitches:
        on = dt > glitch.t
        d = np.where(on, dt - glitch.t, 0.0)
        recovery = glitch.df_transient * glitch.tau * (1 - np.exp(-d / glitch.tau))
        expected += 2 * math.pi * on * (glitch.df_permanent * d + recovery + glitch.df1dot_permanent * d**2 / 2)
    assert np.allclose(phase, expected, rtol=0, atol=1e-6)
