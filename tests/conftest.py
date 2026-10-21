from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cwglitch import Config, Glitch, Signal

TREF = 1372426000.0
TSFT = 1800.0
H0 = 1e-25
DISTANCE_KPC = 3.4
MOMENT_OF_INERTIA = 1e38


@pytest.fixture
def config_dict(tmp_path: Path) -> dict[str, Any]:
    return {
        "output": {"dir": str(tmp_path / "data"), "params_file": "signal_params.json", "sft_label": "simCW{idx}"},
        "seed": 0,
        "n_cpu": 1,
        "source": {
            "n_signals": 3,
            "h0": H0,
            "distance_kpc": DISTANCE_KPC,
            "moment_of_inertia": MOMENT_OF_INERTIA,
            "freq": [100.0, 100.0],
            "f1dot": [-1.056e-8, -1.056e-8],
            "f2dot": [0.0, 0.0],
            "f3dot": [0.0, 0.0],
            "f4dot": [0.0, 0.0],
            "phi0": [0.0, 0.0],
            "psi": [0.785, 0.785],
            "cosi": [1.0, 1.0],
            "alpha": 6.124,
            "delta": 1.026,
        },
        "glitch": {
            "n_per_signal": 1,
            "tglitch": [0.4, 0.4],
            "dnu_nu": [1e-6, 1e-6],
            "dnu1_nu1": [1e-3, 1e-3],
            "healing_factor": [0.7, 0.7],
            "tau": [864000.0, 864000.0],
            "evenly_spaced_tglitch": False,
        },
        "observation": {
            "ifos": ["H1", "L1"],
            "tref": TREF,
            "Tsft": TSFT,
            "noise": {
                "kind": "gaussian",
                "sqrtSX": 1e-22,
                "timestamps": str(tmp_path / "timestamps.json"),
            },
            "dt_wf": 5.0,
            "window_type": "tukey",
            "window_param": 0.001,
        },
    }


@pytest.fixture
def config(config_dict: dict[str, Any]) -> Config:
    return Config.model_validate(config_dict)


@pytest.fixture
def signal() -> Signal:
    return Signal(
        idx=0,
        f0=100.0,
        f1dot=-1.06e-8,
        phi0=0.3,
        psi=0.785,
        cosi=0.5,
        alpha=6.124,
        delta=1.026,
        glitches=[Glitch(tglitch=0.5, dnu_nu=1e-6, dnu1_nu1=1e-3, healing_factor=0.7, tau=864000.0)],
    )


@pytest.fixture
def timestamps() -> np.ndarray:
    return TREF + TSFT * np.arange(6) * 1.5  # gappy
