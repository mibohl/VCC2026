# experiments

The playground. One folder per question; each has a README stating the question, the scripts and where they
write. Experiments read the summaries in `data/`, import pipeline code from `data_processing/src/` and reusable
code from `tools/`, and write to `results/<topic>/`.

| Topic | Question |
| --- | --- |
| [`mergeability/`](mergeability/) | Do the same targets have the same effects across contexts, and can contexts be pooled? |
| [`coverage/`](coverage/) | How well do the summarized contexts cover a challenge panel's genes and targets? |
| [`qc/`](qc/) | Do our summaries agree with the authors' published effects and with themselves? |
| [`h1_baselines/`](h1_baselines/) | Held-out baselines for the internal H1 evaluation |

## Starting a new experiment

1. Create `experiments/<topic>/` with a README.
2. Locate the project root from the script: `ROOT = Path(__file__).resolve().parents[2]`.
3. Default outputs to `ROOT / "results/<topic>"` and job logs to `logs/`.
4. If another experiment starts to depend on a function, move it to `tools/`.
