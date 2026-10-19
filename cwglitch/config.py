from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpindownFromAge(Strict):
    kind: Literal["age"]
    years: float = Field(gt=0)  # f1dot = -f0 / age


class SpindownRange(Strict):
    kind: Literal["f1dot"]
    range: tuple[float, float]


class SourceConfig(Strict):
    n_signals: int = Field(gt=0)
    h0: float = Field(gt=0)
    freq: tuple[float, float]
    spindown: SpindownFromAge | SpindownRange = Field(discriminator="kind")
    f2dot: tuple[float, float]
    f3dot: tuple[float, float]
    f4dot: tuple[float, float]
    phi0: tuple[float, float]
    psi: tuple[float, float]
    cosi: tuple[float, float]
    alpha: float
    delta: float


class GlitchConfig(Strict):
    n_per_signal: int = Field(ge=0)
    tglitch: tuple[float, float]  # fraction of the observation span
    dnu_nu: tuple[float, float]
    dnu1_nu1: tuple[float, float]
    Q: tuple[float, float]
    tau: tuple[float, float]  # seconds
    evenly_spaced_tglitch: bool  # first glitch of each signal on an even grid over the tglitch range

    @model_validator(mode="after")
    def _check(self) -> "GlitchConfig":
        for name in ("tglitch", "Q"):
            lo, hi = getattr(self, name)
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"'{name}' range must satisfy 0 <= lo <= hi <= 1.")
        if self.tau[0] <= 0:
            raise ValueError("'tau' must be positive.")
        return self


class GaussianNoise(Strict):
    kind: Literal["gaussian"]
    sqrtSX: float = Field(gt=0)


class RealData(Strict):
    kind: Literal["real"]
    sft_dir: Path  # inject into the SFTs found here


class ObservationConfig(Strict):
    ifos: list[str] = Field(min_length=1)
    tref: float
    Tsft: float = Field(gt=0)
    timestamps_file: Path  # CSV, one column per IFO of SFT start times (GPS), NaN padded
    noise: GaussianNoise | RealData = Field(discriminator="kind")
    dt_wf: float = Field(gt=0)
    window_type: str
    window_param: float


class Config(Strict):
    label: str
    out_dir: Path
    seed: int
    n_cpu: int = Field(gt=0)
    source: SourceConfig
    glitch: GlitchConfig
    observation: ObservationConfig

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.label

    @property
    def params_file(self) -> Path:
        return self.run_dir / "signal_params.json"


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
