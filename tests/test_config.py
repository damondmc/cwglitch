from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from cwglitch import Config, read_params, sample_signals, write_params


def test_sft_label_checked(config_dict: dict[str, Any]) -> None:
    config_dict["output"]["sft_label"] = "sim_{idx}"  # underscore is not allowed in SFT descriptions
    with pytest.raises(ValidationError, match="alphanumeric"):
        Config.model_validate(config_dict)
    config_dict["output"]["sft_label"] = "simCW"  # no index with several signals
    with pytest.raises(ValidationError, match="idx"):
        Config.model_validate(config_dict)


def test_healing_factor_range_checked(config_dict: dict[str, Any]) -> None:
    config_dict["glitch"]["healing_factor"] = [0.0, 2.0]
    with pytest.raises(ValidationError, match="healing_factor"):
        Config.model_validate(config_dict)


def test_sampling_is_reproducible_and_round_trips(config: Config, tmp_path: Path) -> None:
    a = sample_signals(config, np.random.default_rng(config.seed))
    b = sample_signals(config, np.random.default_rng(config.seed))
    assert a == b
    assert len(a) == config.source.n_signals
    assert all(len(s.glitches) == 1 for s in a)
    path = tmp_path / "p.json"
    write_params(a, path)
    assert read_params(path) == a


def test_evenly_spaced_tglitch(config_dict: dict[str, Any]) -> None:
    config_dict["glitch"]["tglitch"] = [0.0, 1.0]
    config_dict["glitch"]["evenly_spaced_tglitch"] = True
    config_dict["glitch"]["n_per_signal"] = 3
    signals = sample_signals(Config.model_validate(config_dict), np.random.default_rng(0))
    epochs = [[g.tglitch for g in s.glitches] for s in signals]
    assert np.allclose(epochs, [[0.0, 0.5, 1.0]] * len(signals))
