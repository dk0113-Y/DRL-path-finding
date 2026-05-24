# Final Probe Discrepancy Interpretation

## Scope

This is a read-only post-hoc audit over existing training snapshots, final probe CSVs, and the final probe run manifest. No training or final probe episodes were rerun.

## Main Finding

Among reward-comparable methods A and F1, A ranks #1 by training recent reward but #2 by final probe reward. The reversal is explained by A degrading from its training recent window on the fixed final-probe seed set, while F1 improves on the same final-probe seed set.

| metric | A_train | A_final | F1_train | F1_final | A_final_minus_train | F1_final_minus_train |
| --- | --- | --- | --- | --- | --- | --- |
| success_rate | 0.8799999952 | 0.7900000215 | 0.7900000215 | 0.8799999952 | -0.08999997377 | 0.08999997377 |
| coverage | 0.9401469231 | 0.9232640266 | 0.9372969866 | 0.9358919263 | -0.01688289642 | -0.001405060291 |
| reward | 155.0739746 | 145.8809204 | 151.3206177 | 154.3789978 | -9.193054199 | 3.058380127 |
| episode_length | 335.7000122 | 358.6900024 | 350.3900146 | 323.1199951 | 22.98999023 | -27.27001953 |
| timeout | 0.1199999973 | 0.2099999934 | 0.2099999934 | 0.1199999973 | 0.08999999613 | -0.08999999613 |

## Reward Comparability

A and F1 keep the same reward formula, so their reward values can be compared directly. R1-R5 intentionally remove or alter reward penalty terms, so R-group reward is not directly comparable with A. Use R-group coverage, success, length, timeout, and paired seed behavior for cross-method conclusions; treat R reward as within-method diagnostic context.

## Paired Seed Summary

Diffs are A minus comparator on identical episode seeds. For coverage and success, higher is better; for length and timeout, lower is better.

| comparison | episodes | coverage_mean | coverage_A_better/tie/worse | success_mean | success_A_better/tie/worse | length_mean | length_A_better/tie/worse | timeout_mean | timeout_A_better/tie/worse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_vs_F1 | 100 | -0.012628 | 41/6/53 | -0.09 | 9/73/18 | 35.57 | 45/3/52 | 0.09 | 9/73/18 |
| A_vs_R1 | 100 | -0.014549 | 41/6/53 | -0.04 | 11/74/15 | 14.73 | 46/6/48 | 0.04 | 11/74/15 |
| A_vs_R2 | 100 | -0.021058 | 40/10/50 | -0.06 | 9/76/15 | 21.97 | 42/6/52 | 0.06 | 9/76/15 |
| A_vs_R3 | 100 | -0.017628 | 43/6/51 | -0.02 | 12/74/14 | 10.28 | 45/8/47 | 0.02 | 12/74/14 |
| A_vs_R4 | 100 | -0.005152 | 46/7/47 | -0.02 | 14/70/16 | 13.54 | 46/5/49 | 0.02 | 14/70/16 |
| A_vs_R5 | 100 | 0.06835 | 67/6/27 | 0.39 | 45/49/6 | -99.97 | 61/15/24 | -0.39 | 45/49/6 |

## Config And Checkpoint Audit

Evaluation-critical config consistency across A/F1/R1-R5: True. All load_state_dict missing/unexpected key lists are empty: True. The final probe seed_base and episode count are shared by construction from run_manifest: seed_base=20261323, episodes=100.

## Interpretation

The audit does not point to a checkpoint loading mismatch, model factory mismatch, state adapter status failure, or evaluation-critical config drift as the cause of A losing the final probe. The evidence instead points to seed-set generalization or train-window selection variance: A's recent training window is strong, but on the final probe seed set it has lower success and coverage, longer trajectories, more repeated visits, and a higher timeout rate than F1.

## Generated Files

- training_vs_final_probe_comparison.csv
- paired_seed_comparison.csv
- config_audit.json
- discrepancy_interpretation.md
