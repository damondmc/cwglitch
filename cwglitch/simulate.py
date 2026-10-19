import math
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Any

import lalpulsar
import numpy as np
import pandas as pd
from lalpulsar import simulateCW
from tqdm import tqdm

from cwglitch.config import Config
from cwglitch.params import Signal, read_params
from cwglitch.waveform import signal_waveform

Timestamps = dict[str, np.ndarray]

# SWIG bindings ship no stubs, so type checkers cannot see lalpulsar's attributes.
lp: Any = lalpulsar


def load_timestamps(path: Path, ifos: list[str]) -> Timestamps:
    df = pd.read_csv(path)
    missing = [ifo for ifo in ifos if ifo not in df.columns]
    if missing:
        raise ValueError(f"Timestamps file {path} has no column for {missing}.")
    return {ifo: df[ifo].dropna().to_numpy(dtype=float) for ifo in ifos}


def _sft_spec(path: Path, cfg: Config, sig: Signal) -> Any:
    spec = lp.SFTFilenameSpec()
    spec.path = str(path)
    spec.window_type = cfg.observation.window_type
    spec.window_param = cfg.observation.window_param
    spec.privMisc = f"simCW{sig.idx}"
    return spec


def simulate_signal(sig: Signal, cfg: Config, timestamps: Timestamps) -> None:
    obs = cfg.observation
    tstart = min(float(timestamps[ifo][0]) for ifo in obs.ifos)
    tend = max(float(timestamps[ifo][-1]) for ifo in obs.ifos)
    Tobs = tend - tstart

    wf, glitches = signal_waveform(sig, cfg.source.h0, tstart, Tobs, obs.tref)

    # Band wide enough for spindown, glitch jumps, Doppler and a small buffer.
    df_glitch = sum(abs(glitch.df_p) + abs(glitch.df_t) + abs(glitch.df1_p) * Tobs for glitch in glitches)
    df = abs(sig.f1) * Tobs + df_glitch + 1.05e-4 * sig.f0 + 0.01
    fmin = float(math.floor(sig.f0 - df))
    fmax = float(math.ceil(sig.f0 + df))
    fband = fmax - fmin

    out_dir = cfg.run_dir / f"simCW{sig.idx}"
    tmp_dir = cfg.run_dir / "tmp" / f"simCW{sig.idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Signal {sig.idx}] Band [{fmin}, {fmax}] Hz, df={df:.2e}", flush=True)

    # Untyped: lalpulsar annotates its numeric defaults as int, the floats are fine at runtime.
    sft_kwargs: dict[str, Any] = dict(
        fmax=fmax,
        fmin=fmin,
        Tsft=obs.Tsft,
        noise_sqrt_Sh=obs.noise.sqrtSX if obs.noise.kind == "gaussian" else 0.0,
        window=obs.window_type,
        window_param=obs.window_param,
        comment=f"simCW{sig.idx}",
        out_dir=str(tmp_dir),
    )

    for ifo in obs.ifos:
        ts = timestamps[ifo]
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Signal {sig.idx}] {ifo}: simulating {len(ts)} SFTs", flush=True)
        t0 = time.time()
        for i, t in enumerate(ts):
            sim = simulateCW.CWSimulator(
                obs.tref, float(t), obs.Tsft, wf, obs.dt_wf, sig.phi0, sig.psi, sig.alpha, sig.delta, ifo
            )
            for _ in sim.write_sft_files(**sft_kwargs):
                pass
            if (i + 1) % 3000 == 0:
                print(f"[Signal {sig.idx}] {ifo}: {i + 1}/{len(ts)} SFTs", flush=True)
        print(f"[Signal {sig.idx}] {ifo}: generated in {time.time() - t0:.1f}s, merging", flush=True)

        sim_sfts = lp.LoadSFTs(lp.SFTdataFind(f"{tmp_dir}/*.sft", None), -1, -1)
        lp.SFTVectorResizeBand(sim_sfts, fmin, fband)

        if obs.noise.kind == "real":
            constraints = lp.SFTConstraints()
            constraints.detector = ifo
            pattern = str(obs.noise.sft_dir / f"{ifo[0]}-*.sft")
            data_sfts = lp.LoadSFTs(lp.SFTdataFind(pattern, constraints), fmin, fmax)
            real_ts = np.array([float(data_sfts.data[k].epoch) for k in range(data_sfts.length)])
            if len(real_ts) != len(ts) or not np.allclose(real_ts, ts, atol=1.0):
                raise ValueError(
                    f"Timestamp mismatch for {ifo}: {len(real_ts)} real SFTs at {pattern} "
                    f"vs {len(ts)} timestamps in {obs.timestamps_file}."
                )
            lp.SFTVectorResizeBand(data_sfts, fmin, fband)
            lp.SFTVectorAdd(data_sfts, sim_sfts)
        else:
            data_sfts = sim_sfts

        lp.WriteSFTVector2StandardFile(data_sfts, _sft_spec(out_dir, cfg, sig), "simCW", True)
        shutil.rmtree(tmp_dir)

    print(f"[Signal {sig.idx}] done", flush=True)


def _worker(args: tuple[Signal, Config, Timestamps]) -> None:
    simulate_signal(*args)


def run_simulation(cfg: Config) -> None:
    signals = read_params(cfg.params_file)
    timestamps = load_timestamps(cfg.observation.timestamps_file, cfg.observation.ifos)
    for ifo, ts in timestamps.items():
        print(f"{ifo}: {len(ts)} timestamps")
    print(f"{len(signals)} signals, {cfg.n_cpu} CPUs")

    t0 = time.time()
    jobs = [(s, cfg, timestamps) for s in signals]
    with mp.Pool(cfg.n_cpu) as pool:
        list(tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc="Simulating"))
    shutil.rmtree(cfg.run_dir / "tmp", ignore_errors=True)
    print(f"Done in {time.time() - t0:.1f}s")
