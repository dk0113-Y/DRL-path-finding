# A Epsilon Final Probe Diagnostic

- This diagnostic uses epsilon-greedy evaluation and is not a replacement for the official greedy final probe.
- The official final probe remains experiment_records/final_probe/.
- The train-seed greedy diagnostic remains experiment_records/final_probe_train_seed_diagnostic/.
- If epsilon=0.04 improves A on the official final seed block, then greedy determinization likely contributes to A's final-probe drop.
- If epsilon=0.04 does not improve A, the remaining likely causes are endpoint checkpoint quality, last-vs-peak mismatch, or train-window/checkpoint mismatch.
- Epsilon diagnostic should not be used as the main SCI result unless all learning methods are evaluated with the same epsilon protocol.

## Official Seed Block Deltas

| metric | epsilon_minus_official_greedy | epsilon_minus_train_recent |
| --- | ---: | ---: |
| success_rate | 0.04999995231628418 | -0.04000002145767212 |
| coverage | 0.021715998649597168 | 0.004833102226257324 |
| reward | 7.0114593505859375 | -2.1815948486328125 |
| episode_length | 6.639984130859375 | 29.629974365234375 |
| repeat_visit_ratio | -0.002155095338821411 | 0.012970462441444397 |
| timeout_flag_or_rate | -0.04999999701976776 | 0.03999999910593033 |

## Train Seed Block Deltas

| metric | epsilon_minus_train_seed_greedy | epsilon_minus_train_recent |
| --- | ---: | ---: |
| success_rate | 0.09999996423721313 | 0.009999990463256836 |
| coverage | 0.02249598503112793 | 0.0023630857467651367 |
| reward | 9.486892700195312 | -0.1491851806640625 |
| episode_length | -5.09002685546875 | 3.90997314453125 |
| repeat_visit_ratio | -0.020075172185897827 | 0.009715721011161804 |
| timeout_flag_or_rate | -0.09999999403953552 | -0.009999997913837433 |

Positive deltas mean the epsilon diagnostic value is higher than the reference row.
