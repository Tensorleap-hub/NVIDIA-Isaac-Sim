# Mean/Std Experiments

This experiment set is the mean/std counterpart to `experiment_second_order/base`.

Use it with:

```bash
python.sh palletjack_sdg/standalone_palletjack_sdg_mean_std.py \
  --config palletjack_sdg/experiments/experiment_mean_std/base/exp15_realistic_mixed_deployment.yaml
```

Notes:
- These configs extend `palletjack_sdg/sdg_config_mean_std.yaml`.
- The initial presets were converted from the existing min/max base presets with:
  - `mean = (min + max) / 2`
  - `std = (max - min) / sqrt(12)`
- That preserves the mean and variance of the original uniform ranges while switching the runner to normal sampling.
