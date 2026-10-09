# EVALUATION — how to reproduce every number ARGUS claims

Every numeric claim in the documentation must be reproducible with the
commands below. Numbers that are not yet reproducible are written as
**"not yet measured"** — never estimated, never copied from a scratch run.

Plan of record: [docs/plan/world-model-core.md](plan/world-model-core.md).

## 1. Test count (quoted in README.md and argus/README.md)

```bash
cd argus
python -m pytest tests --collect-only -q --no-header -p no:cacheprovider | tail -1
# -> NNN tests collected
```

## 2. Test results (0 failed is the acceptance bar)

```bash
cd argus
python -m pytest tests -q --no-header -p no:cacheprovider
# torch-dependent tests skip when PyTorch is absent; the EICAR test skips if
# host antivirus intercepts the fixture.
```

## 3. Feature count (the "46-column schema" claim)

```bash
cd argus
python -c "from ml.world_model.features import WORLD_MODEL_FEATURES as F; print(len(F))"
# -> 46
```

## 4. Model loss weights & architecture (README "Combined Loss" / dual-engine table)

```bash
cd argus
grep -n "alpha\|beta\|gamma\|delta\|d_model\|n_heads\|n_layers" ml/world_model/model.py | head
# -> defaults: alpha=0.25, beta=0.30, gamma=0.25, delta=0.20,
#    d_model=128, n_heads=4, n_layers=4
```

## 5. CIC-IDS-2017 dataset provenance (row counts, hashes, pinned revision)

```bash
python -m json.tool argus/data/cicids2017_provenance.json | head -40
# -> source repo + revision, per-file sha256 + rows (committed file)
```

## 6. Binned-pipeline manifest (rows kept, labels, hosts, sequences/horizon)

```bash
cd argus
python scripts/prepare_cicids2017.py --check
# -> prints the manifest for the local data/processed/ build (data is
#    gitignored; rebuild it first with `python scripts/prepare_cicids2017.py`)
```

## 7. Served engine & model status (what the deployment reports)

```bash
curl -s https://argusforensics.onrender.com/api/health
# -> {"status": "ok", "rf_model_trained": ..., "world_model_trained": ...,
#     "torch_available": ...}
curl -s -X POST https://argusforensics.onrender.com/api/forecast \
  -H 'content-type: application/json' -d '{"k_steps":4}'
# -> response contains "engine": "neural" | "heuristic"
```

## 8. Benchmark numbers (Accuracy / Macro-F1 / AUC)

Committed: `ml/pretrained/benchmark_results.json` (WP6, **synthetic** split —
400 train / 100 test, `leakage_detected: false`, provenance block inside).
Verify by reading the file, or re-run the harness:

```bash
cd argus
python -m ml.world_model.benchmark --dataset synthetic --sequences 500
```

Any figure not in that JSON (real CIC-IDS-2017 PR-AUC lift, FPR on minority
classes, early-warning lead time on real data) is **not yet measured** — the
training wiring for the binned real data is pending WP6
(`docs/plan/world-model-core.md`).

## 9. Stale-claim audit (run before any docs commit)

```bash
grep -rn "99.97\|0.983\|612,079\|38-Feature\|38 features\|Dirichlet\|embedded" \
  README.md argus/README.md argus/dashboard/static/index.html \
  || echo "no stale claims"
```

## 10. Full local smoke test (dashboard endpoints)

```bash
cd argus
python -m uvicorn dashboard.app:app --port 8123 &
curl -s localhost:8123/api/health
curl -s localhost:8123/api/world-model/status
curl -s localhost:8123/api/benchmark
```
