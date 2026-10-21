import math

import numpy as np
import pytest
from lalpulsar import simulateCW

from cwglitch import SFTGenerator, Signal, frequency_band, signal_waveform
from cwglitch.waveform import Waveform

from .conftest import DISTANCE_KPC, H0, MOMENT_OF_INERTIA, TREF, TSFT


def _generator(signal: Signal, timestamps: np.ndarray) -> tuple[SFTGenerator, float, float, Waveform]:
    tobs = timestamps[-1] - timestamps[0]
    waveform, glitches = signal_waveform(signal, H0, DISTANCE_KPC, MOMENT_OF_INERTIA, timestamps[0], tobs, TREF)
    fmin, fmax = frequency_band(signal, glitches, timestamps[0], timestamps[-1] + TSFT, TREF)
    generator = SFTGenerator(
        signal, waveform, "H1", TREF, timestamps[0], timestamps[-1] + TSFT, 5.0, TSFT, fmin, fmax, "tukey", 0.001
    )
    return generator, fmin, fmax, waveform


def test_frequency_band_contains_signal_even_with_tref_outside_data(signal: Signal) -> None:
    year = 365.25 * 86400.0
    tstart, tend = TREF + year, TREF + 1.5 * year  # data starts a year after tref
    _, glitches = signal_waveform(signal, H0, DISTANCE_KPC, MOMENT_OF_INERTIA, tstart, tend - tstart, TREF)
    fmin, fmax = frequency_band(signal, glitches, tstart, tend, TREF)
    assert fmin == math.floor(fmin) and fmax == math.ceil(fmax)
    doppler = 1e-4 * signal.f0
    for t in (tstart, tend):
        observed = signal.f0 + signal.f1dot * (t - TREF)  # drifted by |f1dot| * 1.5 yr = 0.5 Hz at the end
        assert fmin < observed - doppler and observed + doppler < fmax


def test_sft_headers(signal: Signal, timestamps: np.ndarray) -> None:
    generator, fmin, fmax, _ = _generator(signal, timestamps)
    sfts = generator.sft_vector(timestamps, 0.0, np.random.default_rng(0))
    assert sfts.length == len(timestamps)
    for k, timestamp in enumerate(timestamps):
        sft = sfts.data[k]
        assert sft.name == "H1"
        assert float(sft.epoch) == timestamp
        assert sft.f0 == fmin
        assert sft.deltaF == pytest.approx(1 / TSFT)
        assert sft.data.length == round((fmax - fmin) * TSFT) + 1


def test_signal_peaks_at_expected_frequency(signal: Signal, timestamps: np.ndarray) -> None:
    generator, _, _, _ = _generator(signal, timestamps)
    sfts = generator.sft_vector(timestamps, 0.0, np.random.default_rng(0))
    for k, timestamp in enumerate(timestamps):
        sft = sfts.data[k]
        peak = sft.f0 + np.argmax(np.abs(sft.data.data)) * sft.deltaF
        expected = signal.f0 + signal.f1dot * (timestamp + TSFT / 2 - TREF)
        assert peak == pytest.approx(expected, abs=1.1e-4 * signal.f0 + 2 / TSFT)  # Doppler plus two bins


def test_noise_level(signal: Signal, timestamps: np.ndarray) -> None:
    generator, _, _, _ = _generator(signal, timestamps)
    sqrt_sx = 1e-22
    sfts = generator.sft_vector(timestamps, sqrt_sx, np.random.default_rng(0))
    bins = np.concatenate([np.array(sfts.data[k].data.data)[200:-200] for k in range(sfts.length)])
    assert np.std(bins) == pytest.approx(sqrt_sx * math.sqrt(TSFT / 2), rel=0.03, abs=0)


def test_noise_is_reproducible(signal: Signal, timestamps: np.ndarray) -> None:
    generator, _, _, _ = _generator(signal, timestamps)
    first = generator.sft_vector(timestamps, 1e-22, np.random.default_rng(7))
    same_seed = generator.sft_vector(timestamps, 1e-22, np.random.default_rng(7))
    other_seed = generator.sft_vector(timestamps, 1e-22, np.random.default_rng(8))
    assert np.array_equal(first.data[0].data.data, same_seed.data[0].data.data)
    assert not np.array_equal(first.data[0].data.data, other_seed.data[0].data.data)


def test_matches_cwsimulator(signal: Signal) -> None:
    """Same bins as lalpulsar's reference implementation (which samples at 2*fmax and reloads ephemerides)."""
    timestamps = np.array([TREF + 20 * TSFT, TREF + 40 * TSFT])
    generator, fmin, fmax, waveform = _generator(signal, timestamps)
    sfts = generator.sft_vector(timestamps, 0.0, np.random.default_rng(0))

    def scalar_waveform(dt: float) -> tuple[float, float, float]:
        phase, aplus, across = waveform(np.array([dt]))
        return float(phase[0]), float(aplus[0]), float(across[0])

    for k, timestamp in enumerate(timestamps):
        simulator = simulateCW.CWSimulator(
            TREF,
            float(timestamp),
            TSFT,
            scalar_waveform,
            5.0,
            signal.phi0,
            signal.psi,
            signal.alpha,
            signal.delta,
            "H1",
        )
        reference = next(simulator.get_sfts(fmax=fmax, fmin=fmin, Tsft=TSFT, window="tukey", window_param=0.001))[0]
        reference_bins = np.array(reference.data.data)
        bins = np.array(sfts.data[k].data.data)[: reference_bins.size]
        assert np.allclose(bins, reference_bins, rtol=1e-2, atol=1e-3 * np.abs(reference_bins).max())
