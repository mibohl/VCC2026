# tools

Reusable components: code that experiments import or call, kept stable and documented. Each tool has its
own folder and README.

| Tool | Purpose |
| --- | --- |
| [`cell_generator/`](cell_generator/) | `vcc_decouple.py`: synthetic perturbed cells whose pseudobulk and per-cell average carry two different fold-change vectors |
| [`h1_evaluator/`](h1_evaluator/) | Internal all-300 H1 evaluation with cell-eval2 (block DE, global PDS/MSE) plus the prediction and reference helpers it needs |

To import a tool from an experiment, add its folder to `sys.path` relative to the project root:

```python
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/cell_generator"))
import vcc_decouple as vd
```

Move code here from `experiments/` once a second analysis depends on it.
