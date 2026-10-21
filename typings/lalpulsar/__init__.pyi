# Partial stubs for the SWIG module `lalpulsar`: only what cwglitch uses. Signatures follow the SWIG docstrings.

from collections.abc import Sequence

from lal import (
    COMPLEX8FrequencySeries,
    Detector,
    LIGOTimeGPS,
    REAL4TimeSeries,
    REAL4TimeVectorSeries,
    REAL8TimeSeries,
)

class COMPLEX8FrequencySeriesVector:
    """An SFT vector."""

    length: int
    data: Sequence[COMPLEX8FrequencySeries]

class EphemerisData: ...

class SkyPosition:
    system: int
    longitude: float
    latitude: float

class PulsarCoherentGW:
    def __init__(self) -> None: ...
    position: SkyPosition
    psi: float
    phi: REAL8TimeSeries
    a: REAL4TimeVectorSeries

class PulsarDetectorResponse:
    def __init__(self) -> None: ...
    site: Detector
    ephemerides: EphemerisData
    heterodyneEpoch: LIGOTimeGPS | float

class LIGOTimeGPSVector:
    length: int
    deltaT: float
    data: Sequence[LIGOTimeGPS]

class SFTDescriptor:
    header: COMPLEX8FrequencySeries
    numBins: int
    version: int
    window_type: str
    window_param: float
    comment: str

class SFTCatalog:
    def __init__(self) -> None: ...
    length: int
    data: Sequence[SFTDescriptor]

class SFTConstraints:
    def __init__(self) -> None: ...
    detector: str
    minStartTime: LIGOTimeGPS | float
    maxStartTime: LIGOTimeGPS | float

class SFTFilenameSpec:
    def __init__(self) -> None: ...
    path: str
    window_type: str
    window_param: float
    privMisc: str

def InitBarycenter(earthEphemerisFile: str, sunEphemerisFile: str) -> EphemerisData: ...
def FindCWDetector(name: str, exactMatch: bool) -> tuple[str, int]: ...
def PulsarSimulateCoherentGW(
    output: REAL4TimeSeries, CWsignal: PulsarCoherentGW, detector: PulsarDetectorResponse
) -> int: ...
def AddGaussianNoise(inSeries: REAL4TimeSeries, sigma: float, seed: int) -> int: ...
def MakeTimestamps(tStart: LIGOTimeGPS | float, Tspan: float, Tsft: float, Toverlap: float) -> LIGOTimeGPSVector: ...
def MakeSFTsFromREAL8TimeSeries(
    timeseries: REAL8TimeSeries, timestamps: LIGOTimeGPSVector, windowType: str, windowParam: float
) -> COMPLEX8FrequencySeriesVector: ...
def CreateSFTVector(numSFTs: int, numBins: int) -> COMPLEX8FrequencySeriesVector: ...
def SFTdataFind(file_pattern: str, constraints: SFTConstraints | None) -> SFTCatalog: ...
def LoadSFTs(catalog: SFTCatalog, fMin: float, fMax: float) -> COMPLEX8FrequencySeriesVector: ...
def SFTVectorResizeBand(SFTs: COMPLEX8FrequencySeriesVector, f0: float, Band: float) -> int: ...
def SFTVectorAdd(a: COMPLEX8FrequencySeriesVector, b: COMPLEX8FrequencySeriesVector) -> int: ...
def WriteSFTVector2StandardFile(
    sftVect: COMPLEX8FrequencySeriesVector, SFTfnspec: SFTFilenameSpec, SFTcomment: str, merged: bool
) -> None: ...
def ReadTimestampsFile(fname: str) -> LIGOTimeGPSVector: ...
