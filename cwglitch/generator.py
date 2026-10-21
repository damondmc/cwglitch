"""Heterodyned SFT generation for one signal in one detector."""

import functools
import math

import lal
import lalpulsar as lp
import numpy as np

from cwglitch.params import Signal
from cwglitch.waveform import PhysicalGlitch, Waveform


@functools.cache
def _ephemerides() -> lp.EphemerisData:
    return lp.InitBarycenter("earth00-40-DE405.dat.gz", "sun00-40-DE405.dat.gz")


def frequency_band(
    sig: Signal, glitches: list[PhysicalGlitch], tstart: float, tend: float, tref: float
) -> tuple[float, float]:
    """Integer-Hz band [fmin, fmax] containing the signal between tstart and tend.

    Covers the spin-down drift from tref to the farther end of the data (tref need not lie inside the data),
    the glitch jumps, the Doppler shift and a small buffer.
    """
    span = max(abs(tstart - tref), abs(tend - tref))
    df_spindown = (
        abs(sig.f1dot) * span
        + abs(sig.f2dot) * span**2 / 2.0
        + abs(sig.f3dot) * span**3 / 6.0
        + abs(sig.f4dot) * span**4 / 24.0
    )
    df_glitch = sum(
        abs(glitch.df_permanent) + abs(glitch.df_transient) + abs(glitch.df1dot_permanent) * (tend - tstart)
        for glitch in glitches
    )
    df = df_spindown + df_glitch + 1.05e-4 * sig.f0 + 0.01
    return float(math.floor(sig.f0 - df)), float(math.ceil(sig.f0 + df))


class SFTGenerator:
    def __init__(
        self,
        sig: Signal,
        wf: Waveform,
        ifo: str,
        tref: float,
        tstart: float,
        tend: float,
        dt_wf: float,
        Tsft: float,
        fmin: float,
        fmax: float,
        window_type: str,
        window_param: float,
    ) -> None:
        # Pad by the maximum barycentric delay so the detector-frame signal is defined at the edges.
        pad = 2.0 * lal.AU_SI / lal.C_SI
        t0 = tstart - pad
        n = math.ceil((tend - tstart + 2.0 * pad) / dt_wf)
        dphi, ap, ax = wf((t0 - tref) + dt_wf * np.arange(n))

        phi = lal.CreateREAL8TimeSeries("phi", t0, 0, dt_wf, lal.DimensionlessUnit, n)
        phi.data.data = sig.phi0 + dphi
        amp = lal.REAL4TimeVectorSeries()
        amp.name = "a+,ax"
        amp.epoch = t0
        amp.deltaT = dt_wf
        amp.f0 = 0
        amp.sampleUnits = lal.StrainUnit
        amp.data = lal.CreateREAL4VectorSequence(n, 2)
        amp.data.data = np.column_stack([ap, ax]).astype(np.float32)

        self._source = lp.PulsarCoherentGW()
        self._source.position.system = lal.COORDINATESYSTEM_EQUATORIAL
        self._source.position.longitude = sig.alpha
        self._source.position.latitude = sig.delta
        self._source.psi = sig.psi
        self._source.phi = phi
        self._source.a = amp

        _, site_index = lp.FindCWDetector(ifo, True)
        self._detector = lp.PulsarDetectorResponse()
        self._detector.site = lal.CachedDetectors[site_index]
        self._detector.ephemerides = _ephemerides()

        # Output series f0 = fmin tells SimulateCoherentGW to heterodyne, so 2 * band sampling suffices.
        self._fs = 2.0 * (fmax - fmin)
        nsamp = round(self._fs * Tsft)
        self._h4 = lal.CreateREAL4TimeSeries(ifo, 0, fmin, 1.0 / self._fs, lal.DimensionlessUnit, nsamp)
        self._h8 = lal.CreateREAL8TimeSeries(ifo, 0, fmin, 1.0 / self._fs, lal.DimensionlessUnit, nsamp)
        self._Tsft = Tsft
        self._window_type = window_type
        self._window_param = window_param

    def sft(self, timestamp: float, sqrtSX: float, seed: int) -> lal.COMPLEX8FrequencySeries:
        """One SFT starting at GPS time timestamp, with Gaussian noise of amplitude sqrtSX if > 0."""
        self._h4.epoch = timestamp
        self._h8.epoch = timestamp
        lp.PulsarSimulateCoherentGW(self._h4, self._source, self._detector)
        if sqrtSX > 0:
            lp.AddGaussianNoise(self._h4, math.sqrt(0.5 * self._fs) * sqrtSX, seed)
        self._h8.data.data = self._h4.data.data
        one_sft = lp.MakeTimestamps(timestamp, self._Tsft, self._Tsft, 0)
        return lp.MakeSFTsFromREAL8TimeSeries(self._h8, one_sft, self._window_type, self._window_param).data[0]

    def sft_vector(
        self, timestamps: np.ndarray, sqrtSX: float, rng: np.random.Generator
    ) -> lp.COMPLEX8FrequencySeriesVector:
        """SFTVector with one SFT per start time in timestamps."""
        vec: lp.COMPLEX8FrequencySeriesVector | None = None
        for i, timestamp in enumerate(timestamps):
            s = self.sft(float(timestamp), sqrtSX, int(rng.integers(1, 2**31 - 1)))
            if vec is None:
                vec = lp.CreateSFTVector(len(timestamps), s.data.length)
            dst = vec.data[i]
            dst.name = s.name
            dst.epoch = s.epoch
            dst.f0 = s.f0
            dst.deltaF = s.deltaF
            dst.sampleUnits = s.sampleUnits
            dst.data.data = s.data.data
        if vec is None:
            raise ValueError("No timestamps given.")
        return vec
