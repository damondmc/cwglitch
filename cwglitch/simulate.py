import multiprocessing as mp
import time
from pathlib import Path

import lalpulsar as lp
import numpy as np
from pydantic import TypeAdapter
from tqdm import tqdm

from cwglitch.config import Config
from cwglitch.generator import SFTGenerator, frequency_band
from cwglitch.params import Signal, read_params
from cwglitch.waveform import signal_waveform

TimestampsByIfo = dict[str, np.ndarray]  # IFO -> SFT start times, GPS seconds


def _real_sft_catalog(sft_glob: str, ifo: str) -> lp.SFTCatalog:
    constraints = lp.SFTConstraints()
    constraints.detector = ifo
    catalog = lp.SFTdataFind(sft_glob, constraints)
    if catalog.length == 0:
        raise ValueError(f"No {ifo} SFTs match {sft_glob}.")
    return catalog


def load_timestamps(cfg: Config) -> TimestampsByIfo:
    obs = cfg.observation
    timestamps_by_ifo: TimestampsByIfo = {}
    if obs.noise.kind == "gaussian":
        stamps = TypeAdapter(dict[str, list[float]]).validate_json(obs.noise.timestamps.read_bytes())
        missing = [ifo for ifo in obs.ifos if ifo not in stamps]
        if missing:
            raise ValueError(f"{obs.noise.timestamps} has no timestamps for {missing}.")
        return {ifo: np.array(stamps[ifo]) for ifo in obs.ifos}

    for ifo in obs.ifos:
        catalog = _real_sft_catalog(obs.noise.sft_glob, ifo)
        tsft = 1.0 / catalog.data[0].header.deltaF
        if tsft != obs.Tsft:
            raise ValueError(f"{ifo} SFTs have Tsft={tsft} s but config says {obs.Tsft} s.")
        timestamps_by_ifo[ifo] = np.array([float(catalog.data[k].header.epoch) for k in range(catalog.length)])
    return timestamps_by_ifo


def _sft_spec(path: Path, cfg: Config, sig: Signal) -> lp.SFTFilenameSpec:
    spec = lp.SFTFilenameSpec()
    spec.path = str(path)
    spec.window_type = cfg.observation.window_type
    spec.window_param = cfg.observation.window_param
    spec.privMisc = cfg.sft_label(sig.idx)
    return spec


def simulate_signal(sig: Signal, ifo: str, cfg: Config, timestamps_by_ifo: TimestampsByIfo) -> None:
    """Write the merged SFT file for one signal in one detector."""
    obs = cfg.observation
    tstart = min(float(timestamps_by_ifo[i][0]) for i in obs.ifos)
    tend = max(float(timestamps_by_ifo[i][-1]) for i in obs.ifos)
    Tobs = tend - tstart
    timestamps = timestamps_by_ifo[ifo]

    src = cfg.source
    wf, glitches = signal_waveform(sig, src.h0, src.distance_kpc, src.moment_of_inertia, tstart, Tobs, obs.tref)

    fmin, fmax = frequency_band(sig, glitches, tstart, tend + obs.Tsft, obs.tref)
    fband = fmax - fmin

    out_dir = cfg.sft_dir(sig.idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[Signal {sig.idx}] {ifo}:"
    print(f"{tag} band [{fmin}, {fmax}] Hz, simulating {len(timestamps)} SFTs", flush=True)

    t0 = time.time()
    generator = SFTGenerator(
        sig,
        wf,
        ifo,
        obs.tref,
        tstart,
        tend + obs.Tsft,
        obs.dt_wf,
        obs.Tsft,
        fmin,
        fmax,
        obs.window_type,
        obs.window_param,
    )
    sqrtSX = obs.noise.sqrtSX if obs.noise.kind == "gaussian" else 0.0
    rng = np.random.default_rng([cfg.seed, sig.idx, obs.ifos.index(ifo)])
    sim_sfts = generator.sft_vector(timestamps, sqrtSX, rng)
    lp.SFTVectorResizeBand(sim_sfts, fmin, fband)
    print(f"{tag} generated in {time.time() - t0:.1f}s, merging", flush=True)

    if obs.noise.kind == "real":
        catalog = _real_sft_catalog(obs.noise.sft_glob, ifo)
        header = catalog.data[0].header
        data_fmin = header.f0
        data_fmax = header.f0 + catalog.data[0].numBins * header.deltaF
        if fmin < data_fmin or fmax > data_fmax:
            raise ValueError(
                f"Signal {sig.idx} needs band [{fmin}, {fmax}] Hz but {ifo} SFTs cover [{data_fmin}, {data_fmax}] Hz."
            )
        data_sfts = lp.LoadSFTs(catalog, fmin, fmax)
        lp.SFTVectorResizeBand(data_sfts, fmin, fband)
        lp.SFTVectorAdd(data_sfts, sim_sfts)
    else:
        data_sfts = sim_sfts

    lp.WriteSFTVector2StandardFile(data_sfts, _sft_spec(out_dir, cfg, sig), "simCW", True)
    print(f"{tag} done", flush=True)


def _worker(args: tuple[Signal, str, Config, TimestampsByIfo]) -> None:
    simulate_signal(*args)


def run_simulation(cfg: Config) -> None:
    signals = read_params(cfg.params_file)
    timestamps_by_ifo = load_timestamps(cfg)
    for ifo, timestamps in timestamps_by_ifo.items():
        print(f"{ifo}: {len(timestamps)} SFT start times")
    jobs = [(s, ifo, cfg, timestamps_by_ifo) for s in signals for ifo in cfg.observation.ifos]
    print(f"{len(signals)} signals x {len(cfg.observation.ifos)} IFOs = {len(jobs)} jobs on {cfg.n_cpu} CPUs")

    t0 = time.time()
    with mp.Pool(cfg.n_cpu) as pool:
        list(tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc="Simulating"))
    print(f"Done in {time.time() - t0:.1f}s")
