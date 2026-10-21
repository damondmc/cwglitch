from pathlib import Path

import numpy as np
from pydantic import BaseModel, TypeAdapter

from cwglitch.config import Config


class Glitch(BaseModel):
    tglitch: float  # fraction of the observation span
    dnu_nu: float
    dnu1_nu1: float
    healing_factor: float
    tau: float  # seconds


class Signal(BaseModel):
    idx: int
    f0: float
    f1dot: float
    f2dot: float = 0.0
    f3dot: float = 0.0
    f4dot: float = 0.0
    phi0: float
    psi: float
    cosi: float
    alpha: float
    delta: float
    glitches: list[Glitch] = []


_signal_list = TypeAdapter(list[Signal])


def sample_signals(cfg: Config, rng: np.random.Generator) -> list[Signal]:
    src, gl = cfg.source, cfg.glitch
    n, m = src.n_signals, gl.n_per_signal

    f0 = rng.uniform(*src.freq, n)
    f1dot = rng.uniform(*src.f1dot, n)
    f2dot = rng.uniform(*src.f2dot, n)
    f3dot = rng.uniform(*src.f3dot, n)
    f4dot = rng.uniform(*src.f4dot, n)
    phi0 = rng.uniform(*src.phi0, n)
    psi = rng.uniform(*src.psi, n)
    cosi = rng.uniform(*src.cosi, n)

    # glitch params
    tglitch = rng.uniform(*gl.tglitch, (n, m))
    dnu_nu = rng.uniform(*gl.dnu_nu, (n, m))
    dnu1_nu1 = rng.uniform(*gl.dnu1_nu1, (n, m))
    healing_factor = rng.uniform(*gl.healing_factor, (n, m))
    tau = rng.uniform(*gl.tau, (n, m))
    if m and gl.evenly_spaced_tglitch:
        tglitch[:] = np.linspace(gl.tglitch[0], gl.tglitch[1], m)

    signals: list[Signal] = []
    for i in range(n):
        order = np.argsort(tglitch[i])
        glitches = [
            Glitch(
                tglitch=tglitch[i, j],
                dnu_nu=dnu_nu[i, j],
                dnu1_nu1=dnu1_nu1[i, j],
                healing_factor=healing_factor[i, j],
                tau=tau[i, j],
            )
            for j in order
        ]
        signals.append(
            Signal(
                idx=i,
                f0=f0[i],
                f1dot=f1dot[i],
                f2dot=f2dot[i],
                f3dot=f3dot[i],
                f4dot=f4dot[i],
                phi0=phi0[i],
                psi=psi[i],
                cosi=cosi[i],
                alpha=src.alpha,
                delta=src.delta,
                glitches=glitches,
            )
        )
    return signals


def write_params(signals: list[Signal], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_signal_list.dump_json(signals, indent=2))


def read_params(path: Path) -> list[Signal]:
    return _signal_list.validate_json(path.read_bytes())
