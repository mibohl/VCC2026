# tests

| Script | Purpose |
| --- | --- |
| `smoke.py` | Small synthetic checks (24 genes, 32 controls, 40 perturbed cells), no downloads: `pixi run smoke` |
| `known_answer.py` | Recomputes one real context with SciPy and cell-eval2's DE: `pixi run python data_processing/tests/known_answer.py --context adamson__k562` (needs the raw files) |

The smoke test also imports `tools/h1_evaluator/` and `experiments/mergeability/mergeability.py`, so it
covers the evaluator and the pairwise-agreement code as well.
