import argparse

import numpy as np

from cwglitch.config import load_config
from cwglitch.params import sample_signals, write_params


def gen_params(config_path: str) -> None:
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg.seed)
    signals = sample_signals(cfg, rng)
    write_params(signals, cfg.params_file)
    print(f"Wrote {len(signals)} signals to {cfg.params_file}")


def simulate(config_path: str) -> None:
    from cwglitch.simulate import run_simulation

    run_simulation(load_config(config_path))


def main() -> None:
    parser = argparse.ArgumentParser(prog="cwglitch", description="Simulate continuous-wave signals with glitches.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("gen-params", "Sample signal and glitch parameters to JSON"),
        ("simulate", "Generate SFTs for the signals in the JSON"),
        ("run", "gen-params followed by simulate"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()

    if args.command in ("gen-params", "run"):
        gen_params(args.config)
    if args.command in ("simulate", "run"):
        simulate(args.config)


if __name__ == "__main__":
    main()
