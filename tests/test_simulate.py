import json
from pathlib import Path

import lalpulsar as lp
import numpy as np
import pytest

from cwglitch import Config, sample_signals, write_params
from cwglitch.simulate import load_timestamps, run_simulation

from .conftest import TREF, TSFT


def test_timestamps_required_per_ifo(config: Config, tmp_path: Path) -> None:
    (tmp_path / "timestamps.json").write_text(json.dumps({"H1": [TREF]}))
    with pytest.raises(ValueError, match="L1"):
        load_timestamps(config)


def test_run_simulation_writes_one_merged_file_per_ifo(config: Config, tmp_path: Path) -> None:
    h1_timestamps = TREF + TSFT * np.arange(4)
    l1_timestamps = TREF + TSFT * np.arange(3)
    timestamps_json = {"H1": h1_timestamps.tolist(), "L1": l1_timestamps.tolist()}
    (tmp_path / "timestamps.json").write_text(json.dumps(timestamps_json))
    signals = sample_signals(config, np.random.default_rng(config.seed))
    write_params(signals, config.params_file)

    run_simulation(config)

    for signal in signals:
        sft_dir = config.sft_dir(signal.idx)
        assert [f.name[:5] for f in sorted(sft_dir.glob("*.sft"))] == ["H-4_H", "L-3_L"]
        sfts = lp.LoadSFTs(lp.SFTdataFind(str(sft_dir / "H-*.sft"), None), -1, -1)
        assert sfts.length == 4
        assert [float(sfts.data[k].epoch) for k in range(4)] == list(h1_timestamps)
        first = sfts.data[0]
        band_max = first.f0 + first.deltaF * first.data.length
        for t in (h1_timestamps[0], h1_timestamps[-1] + TSFT):
            observed = signal.f0 + signal.f1dot * (t - TREF)
            assert first.f0 < observed < band_max
