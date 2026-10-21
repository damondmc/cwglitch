# Partial stubs for the SWIG module `lal`: only what cwglitch uses. Signatures follow the SWIG docstrings.

from collections.abc import Sequence

import numpy as np

AU_SI: float
C_SI: float
G_SI: float
PC_SI: float
COORDINATESYSTEM_EQUATORIAL: int

class Unit: ...

DimensionlessUnit: Unit
StrainUnit: Unit

class LIGOTimeGPS:
    def __init__(self, seconds: float = ..., nanoseconds: int = ...) -> None: ...
    def __float__(self) -> float: ...

class Detector:
    location: np.ndarray

CachedDetectors: Sequence[Detector]

class REAL4Vector:
    length: int
    data: np.ndarray

class REAL8Vector:
    length: int
    data: np.ndarray

class COMPLEX8Vector:
    length: int
    data: np.ndarray

class REAL4VectorSequence:
    length: int
    vectorLength: int
    data: np.ndarray

class REAL4TimeSeries:
    name: str
    epoch: LIGOTimeGPS | float
    f0: float
    deltaT: float
    sampleUnits: Unit
    data: REAL4Vector

class REAL8TimeSeries:
    name: str
    epoch: LIGOTimeGPS | float
    f0: float
    deltaT: float
    sampleUnits: Unit
    data: REAL8Vector

class REAL4TimeVectorSeries:
    def __init__(self) -> None: ...
    name: str
    epoch: LIGOTimeGPS | float
    f0: float
    deltaT: float
    sampleUnits: Unit
    data: REAL4VectorSequence

class COMPLEX8FrequencySeries:
    """One SFT."""

    name: str
    epoch: LIGOTimeGPS | float
    f0: float
    deltaF: float
    sampleUnits: Unit
    data: COMPLEX8Vector

def CreateREAL4TimeSeries(
    name: str, epoch: LIGOTimeGPS | float, f0: float, deltaT: float, sampleUnits: Unit, length: int
) -> REAL4TimeSeries: ...
def CreateREAL8TimeSeries(
    name: str, epoch: LIGOTimeGPS | float, f0: float, deltaT: float, sampleUnits: Unit, length: int
) -> REAL8TimeSeries: ...
def CreateREAL4VectorSequence(length: int, veclen: int) -> REAL4VectorSequence: ...
