# h1_baselines

Reference points for the internal H1 evaluation (`tools/h1_evaluator/`): what does a held-out half-split
oracle or the mean response score?

| Script | Purpose | Output |
| --- | --- | --- |
| `h1_heldout_predictions.py` | Per seed: a half-split oracle, the mean response, and the rows of the held-out cells | `results/h1_heldout/seed<N>/{half_oracle,mean_response,holdout}.npz` |

```bash
pixi run python experiments/h1_baselines/h1_heldout_predictions.py --seed 0
sbatch tools/h1_evaluator/evaluate_h1.sbatch --predictions results/h1_heldout/seed0/half_oracle.npz \
    --holdout results/h1_heldout/seed0/holdout.npz --out results/h1_heldout/half_oracle_gain0.25_seed0 --seed 0
```

Earlier runs: `results/h1_oracle/` (in-sample oracle by gain and seed), `results/h1_heldout_pre_zerofix/`
(before held-out fold changes used zero for genes without control counts), `results/h1_bundle_20k/`.
