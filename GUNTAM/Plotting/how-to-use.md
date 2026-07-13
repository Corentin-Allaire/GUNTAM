# GUNTAM.Plotting — How to use

ROOT comparison plotting for `performance_seeding.root` / `performance_finding_ckf.root` /
`performance_finding_ambi.root` outputs. Reads `TEfficiency`/`TProfile` objects with `uproot`
(no PyROOT/ROOT install required) and produces report-ready comparison figures plus CSV/Markdown
summary tables.

## Module layout

| File | Responsibility |
|---|---|
| `Config.py` | `PlottingConfig` dataclass, CLI flags, hybrid JSON+CLI loading (`--config`/`--save_config`) |
| `PlotStyle.py` | Color palette, rcParams, per-quantity label/axis styling, the shared (bug-fixed) drawing routines |
| `RootIO.py` | uproot access: key introspection, Clopper-Pearson efficiency, inverse-variance profile combination |
| `ReportFigure.py` | Entrance script: 2x2 report-ready seeding-vs-finding comparison figure |
| `BatchSweep.py` | Entrance script: sweep an arbitrary set of quantities across an arbitrary set of ROOT files |
| `SummaryTables.py` | CSV/Markdown summary tables written by `BatchSweep` |

`example_config.json` in this directory is the literal `PlottingConfig()` default, serialized —
use it as a schema reference or a starting point for your own `--config` file.

## Quick start

### ReportFigure — one 2x2 comparison figure

`--files` are **directories** (one per dataset/setup), each expected to contain
`performance_seeding.root` plus `performance_finding_ckf.root` or `performance_finding_ambi.root`
(selected by `--compare`):

```bash
python -m GUNTAM.Plotting.ReportFigure \
  --files path/to/datasetA path/to/datasetB \
  --labels "Dataset A" "Dataset B" \
  --quantities trackeff_vs_z0 trackeff_vs_phi \
  --compare seeding-vs-ckf \
  --output_dir plots
```

`--quantities` must be **exactly 2** values (no `all`). The figure is laid out as:
- **rows** = stage — the compare-stage (CKF/Ambiguity resolution) on top, Seeding on the bottom,
  labelled with a rotated side label (like the reference figure's "Phase 1"/"Phase 2")
- **columns** = quantity, in the order you passed them to `--quantities`
- a single shared legend (dataset colors/markers are the same across every panel)

`--compare` is one of `seeding-vs-ckf` / `seeding-vs-ambi`.

### BatchSweep — every quantity across a set of files

`--files` are literal ROOT file paths, all of one "kind" (e.g. all `*_seeding.root`):

```bash
python -m GUNTAM.Plotting.BatchSweep \
  --files path/to/A/performance_seeding.root path/to/B/performance_seeding.root \
  --labels "Dataset A" "Dataset B" \
  --output_dir plots
```

`--quantities` defaults to `all`, which sweeps every 1D `TEfficiency`/`TProfile` key found in the
first `--files` entry (2D efficiency objects, e.g. `trackeff_vs_eta_phi`, are skipped
automatically — they share the same ROOT classname as 1D ones but aren't representable on a
single x-axis). Pass explicit key names to restrict the sweep. One figure is written per
quantity, plus `summary.csv` and `summary.md`.

## Configuration file (`--config` / `--save_config`)

`PlottingConfig` can be loaded from and saved to JSON:

```bash
# save whatever you passed on the CLI (plus unset fields' defaults) to a JSON file
python -m GUNTAM.Plotting.BatchSweep --files a.root b.root --save_config my_run.json

# reload it later, optionally overriding individual fields
python -m GUNTAM.Plotting.BatchSweep --config my_run.json --output_dir other_plots
```

Precedence: dataclass defaults → `--config` JSON (if given) → any CLI flags you also passed
(applied on top). A CLI `--files` **replaces** the JSON's `files` list wholesale — it never
merges/appends. `--labels` length is validated against the **final**, post-override `files`
length, so if you override `--files` you may also need to override `--labels`.

Fields: `files`, `labels`, `compare` (ReportFigure only), `quantities`, `output_dir`,
`output_formats`, `dpi` — see `example_config.json` for the defaults.

## Output artifacts

- One image per quantity (`BatchSweep`) or one 2x2 image per quantity pair (`ReportFigure`),
  in every format listed in `--output_formats` (default `["png"]`).
- `summary.csv` — one flat table: one row per quantity, one column per dataset. Meant for
  further processing, not restructured.
- `summary.md` — two sections:
  1. **`## Overall`** — one row per *metric* (e.g. `trackeff`, `purity`, `nHoles`), combining
     every quantity sharing that metric prefix (`trackeff_vs_eta`, `trackeff_vs_pT`, ...) via an
     inverse-variance-weighted average. Conditional slices (e.g. `trackeff_vs_eta_ptRange_0`,
     restricted to one pT range) are excluded from this combination — a slice's pooled sample
     size is checked against the group's largest re-binning, and anything below 90% of it is
     treated as a slice rather than another estimate of the same overall number
     (`SummaryTables._full_population_keys`).
  2. **`## Full breakdown`** — every individual quantity, including the excluded slices.

## Statistics

- **Efficiency (`TEfficiency`)**: 68% Clopper-Pearson CI, either per-bin
  (`RootIO.read_efficiency`) or pooled across all bins into one number
  (`RootIO.pooled_efficiency`).
- **Profile (`TProfile`)**: inverse-variance-weighted combination across bins
  (`RootIO.inverse_variance_combine` / `RootIO.combine_profile_inverse_variance`), excluding
  non-finite or zero-error bins (an empty TProfile bin reports `value=0, error=0`, which would
  otherwise blow up the weight).
- **`d0` is folded** (`|d0|`, fixed `(0, 10)` mm range) since its sign is an arbitrary side
  convention; **`z0` is left signed and unbounded** since it carries real physical meaning along
  the beam axis.

## Extending quantity labels

Pretty labels/units are composed from two small tables in `PlotStyle.py`, split on the ROOT
key's `_vs_`:

- `AXIS_STYLES` — one entry per x-axis variable (`eta`, `phi`, `pT`, `d0`, `z0`): x-label, x-limit,
  fold-absolute.
- `METRIC_LABELS` — one entry per metric prefix (`trackeff`, `fakeRatio`, `nHoles`, ...): y-label.

Any `<metric>_vs_<axis>` key is composed automatically from whichever of these two tables have a
matching entry; unmatched metrics/axes fall back to the raw key substring rather than crashing,
so a brand-new ROOT key still gets *something* readable without a code change.

## Known limitations

- Quantities using an axis suffix not in `AXIS_STYLES` (e.g. `eta_ptRange_0`) get a readable
  y-label but a raw x-axis label — add an `AXIS_STYLES` entry if you want it prettified.
- A `TProfile` quantity whose bins are genuinely all `value=0, error=0` (not just empty, but a
  real all-zero measurement, e.g. `nSharedHits_vs_*` in the sample data) reports `nan ± nan`,
  since that pattern is indistinguishable from "no entries" using only value+error.
