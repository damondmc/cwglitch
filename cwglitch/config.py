import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(Strict):
    n_signals: int = Field(gt=0)
    h0: float = Field(gt=0)
    distance_kpc: float = Field(gt=0)
    moment_of_inertia: float = Field(gt=0)  # kg m^2
    freq: tuple[float, float]
    f1dot: tuple[float, float]
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
    healing_factor: tuple[float, float]
    tau: tuple[float, float]  # seconds
    evenly_spaced_tglitch: bool

    @model_validator(mode="after")
    def _check(self) -> "GlitchConfig":
        for name in ("tglitch", "healing_factor"):
            lo, hi = getattr(self, name)
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"'{name}' range must satisfy 0 <= lo <= hi <= 1.")
        if self.tau[0] <= 0:
            raise ValueError("'tau' must be positive.")
        return self


class GaussianNoise(Strict):
    kind: Literal["gaussian"]
    sqrtSX: float = Field(gt=0)
    timestamps: Path  # {"H1": [GPS start times], "L1": [...]}


class RealData(Strict):
    kind: Literal["real"]
    sft_glob: str  # SFT files to inject into; IFO is selected from the SFT headers, timestamps come from them


class ObservationConfig(Strict):
    ifos: list[str] = Field(min_length=1)
    tref: float
    Tsft: float = Field(gt=0)
    noise: GaussianNoise | RealData = Field(discriminator="kind")
    dt_wf: float = Field(gt=0)
    window_type: str
    window_param: float


class OutputConfig(Strict):
    dir: Path
    params_file: str
    sft_label: str


class Config(Strict):
    output: OutputConfig
    seed: int
    n_cpu: int = Field(gt=0)
    source: SourceConfig
    glitch: GlitchConfig
    observation: ObservationConfig

    @model_validator(mode="after")
    def _check(self) -> "Config":
        label = self.output.sft_label
        if self.source.n_signals > 1 and "{idx}" not in label:
            raise ValueError("'sft_label' must contain {idx} when n_signals > 1.")
        if not re.fullmatch(r"[A-Za-z0-9]+", label.format(idx=0)):
            raise ValueError("'sft_label' must be alphanumeric: LAL SFT file descriptions allow only A-Za-z0-9.")
        return self

    @property
    def params_file(self) -> Path:
        return self.output.dir / self.output.params_file

    def sft_label(self, idx: int) -> str:
        return self.output.sft_label.format(idx=idx)

    def sft_dir(self, idx: int) -> Path:
        return self.output.dir / self.sft_label(idx)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
