# cwglitch

Simulate continuous gravitational-wave signals from pulsars with glitches, written as SFTs via LALSuite's `CWSimulator`.

## Install

Managed with [uv](https://docs.astral.sh/uv/). Python version is pinned in `.python-version` and dependencies in `uv.lock`.

```
uv sync
```

Without uv: `pip install -e .` into an env that provides `lalpulsar` (the `lalsuite` wheel or a conda env).

## Usage

```
uv run cwglitch gen-params config/example.yaml   # sample parameters -> <out_dir>/<label>/signal_params.json
uv run cwglitch simulate   config/example.yaml   # write SFTs       -> <out_dir>/<label>/simCW<i>/
uv run cwglitch run        config/example.yaml   # both
```

See `config/example.yaml` for all options. Each `[lo, hi]` entry is sampled uniformly per signal.

## Glitch model

Each glitch is parameterised relative to the source: `tglitch` (fraction of the observation span),
`dnu_nu`, `dnu1_nu1`, healing fraction `Q`, and recovery time `tau`. For `t > tglitch`:

```
f(t)  = f0 + (1 - Q) dnu + Q dnu exp(-(t - tglitch) / tau)
f1(t) = f1 + dnu1
```

with `dnu = dnu_nu * f0` and `dnu1 = dnu1_nu1 * f1`. The strain amplitude is scaled as
`h0 * sqrt(|f1(t)| / f(t))` relative to the pre-glitch value.

The glitch parameterisation (permanent step, healing fraction `Q`, recovery time `tau`) follows
Yim & Jones (2020), *Transient gravitational waves from pulsar post-glitch recoveries*,
MNRAS 498, 3138, [doi:10.1093/mnras/staa2534](https://doi.org/10.1093/mnras/staa2534),
[arXiv:2007.05893](https://arxiv.org/abs/2007.05893).
