# Partial stubs for lalpulsar.simulateCW: only what the tests use.

from collections.abc import Callable, Iterator

from lal import COMPLEX8FrequencySeries

class CWSimulator:
    def __init__(
        self,
        tref: float,
        tstart: float,
        Tdata: float,
        waveform: Callable[[float], tuple[float, float, float]],
        dt_wf: float,
        phi0: float,
        psi: float,
        alpha: float,
        delta: float,
        det_name: str,
        earth_ephem_file: str = ...,
        sun_ephem_file: str = ...,
        tref_at_det: bool = ...,
        extra_comment: str | None = ...,
    ) -> None: ...
    def get_sfts(
        self,
        fmax: float,
        Tsft: float,
        noise_sqrt_Sh: float = ...,
        noise_seed: int | None = ...,
        window: str = ...,
        window_param: float = ...,
        prog_bar: object = ...,
        fmin: float = ...,
    ) -> Iterator[tuple[COMPLEX8FrequencySeries, int, int]]: ...
