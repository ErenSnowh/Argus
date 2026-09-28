# ml/pretrained/ - committed model artifacts

This directory is tracked by git, unlike `ml/artifacts/` (gitignored, holds
machine-local runs). What a fresh clone and the SIH evaluator see comes from
here. `dashboard/app.py` resolves benchmark results as `ml/artifacts/` first,
then `ml/pretrained/`.

## Required contents (shipped by WP6)

| File | Purpose |
|---|---|
| `world_model.pt` | Transformer checkpoint: `model_state_dict`, ordered `feature_list`, `n_features`, `bin_seconds`, `W`, `horizons`, `train_mean`, `train_std`, dataset manifest hash, git SHA, UTC timestamp |
| `world_model_metrics.json` | Training metrics written by `ml/world_model/train.py`, with the same `provenance` block |
| `benchmark_results.json` | Output of `argus benchmark` (`ml/world_model/benchmark.py`), with `provenance` |
| `PROVENANCE.md` | Human-readable provenance: dataset, source files + MD5s, row and sequence counts, split protocol, environment (CPU/GPU, measured minutes), and the exact commands that reproduce each file |

## Rules

1. No artifact may live here without a `provenance` block (dataset, files, rows,
   split protocol, git SHA, UTC timestamp). A metric without provenance is not a
   result.
2. If neither `ml/artifacts/` nor this directory holds a result, the API returns
   404 / false. It never substitutes a synthetic or remembered number.
3. Do not hand-edit the JSON files here. Regenerate them with the committed
   commands and re-run `scripts/render_readme_metrics.py`.
4. Until WP6 runs on real data this directory is intentionally empty of
   artifacts, and `/api/benchmark` correctly 404s.

See `docs/plan/world-model-core.md` (D5, WP6).