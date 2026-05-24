# A Train-Seed Final Probe Diagnostic

- This diagnostic uses the training seed base and is not a replacement for the held-out final probe.
- The official held-out final probe remains experiment_records/final_probe/.
- If train-seed greedy probe is closer to train recent-window metrics than held-out final probe, the drop is likely related to seed-distribution shift or held-out generalization.
- If train-seed greedy probe is still much lower than train recent-window metrics, the drop may be related to greedy determinization, endpoint checkpoint quality, or train-window vs checkpoint mismatch.

## Metric Deltas

| metric | train_seed_minus_train_recent | train_seed_minus_official_held_out |
| --- | ---: | ---: |
| success_rate | -0.08999997374232793 | 0.0 |
| coverage | -0.020132899319177278 | -0.0032500028610229492 |
| reward | -9.636077871484389 | -0.443023681640625 |
| episode_length | 9.000000007031247 | -13.989990234375 |
| repeat_visit_ratio | 0.029790893151467907 | 0.014665335416793823 |
| timeout_flag_or_rate | 0.08999999614348908 | 0.0 |

Positive deltas mean the train-seed diagnostic value is higher than the reference row.
