# Training Curve Interpretation

## Scope

This analysis reads existing formal train logs, metric snapshots, final probe summaries, and checkpoint path inventories. It does not rerun training or final probe evaluation, and it does not copy checkpoint files.

B, the classical frontier greedy baseline, is excluded from training curves.

## Reward Comparability

R-group rewards are not directly comparable with A because reward penalty terms are ablated. Use R-group reward curves as within-method diagnostics; compare R-group methods against A primarily through coverage, success, episode length, timeout, and repeat-visit trends.

## Peak Vs Last Summary

| method_id | peak-last coverage | peak-last success | peak-last reward | last not peak |
| --- | --- | --- | --- | --- |
| A | 0.007254064083 | 0.009999990463 | 1.81980896 | True |
| F1 | 0.00760191679 | 0.1099999547 | 5.711715698 | True |
| R1 | 0.02159303427 | 0.06000000238 | 3.988067627 | True |
| R2 | 0.01818996668 | 0.09999996424 | 6.660903931 | True |
| R3 | 0.03135597706 | 0.1599999666 | 10.16012573 | True |
| R4 | 0.01610696316 | 0.2400000095 | 13.03723145 | True |
| R5 | 0.000550031662 | 0.009999990463 | 0.3333129883 | True |

## Late-Stage Degradation Signals

Late-stage rows compare the 400k-500k env-step window with the final logged point. Positive coverage/success drops and positive timeout/length increases indicate that the final logged point is worse than a late-window reference point.

| method_id | coverage drop | success drop | timeout increase | length increase | signal |
| --- | --- | --- | --- | --- | --- |
| A | 0.007254064083 | 0.009999990463 | 0.009999997914 | 11.36001587 | True |
| F1 | 0.00760191679 | 0.1099999547 | 0.109999992 | 30.83001709 | True |
| R1 | 0.01765400171 | 0.06000000238 | 0.06000000238 | 11 | True |
| R2 | 0.01818996668 | 0.08999997377 | 0.09000001848 | 27 | True |
| R3 | 0.03135597706 | 0.1599999666 | 0.1599999964 | 27.63998413 | True |
| R4 | 0.01610696316 | 0.2400000095 | 0.240000017 | 43.91998291 | True |
| R5 | 0.000550031662 | 0.009999990463 | 0.009999990463 | 6.579986572 | True |

## A-Specific Read

A late-stage degradation signal: True. Coverage drop from late-window peak is 0.007254064083, success drop is 0.009999990463, timeout increase from late-window minimum is 0.009999997914, and episode-length increase from late-window minimum is 11.36001587.

## Checkpoint Candidate Audit

| method_id | only last.pt | periodic | best.pt | store export |
| --- | --- | --- | --- | --- |
| A | False | False | False | True |
| C | True | False | False | True |
| D | True | False | False | True |
| E | True | False | False | True |
| F1 | True | False | False | True |
| F2 | True | False | False | True |
| F3 | True | False | False | True |
| F4 | True | False | False | True |
| F5 | True | False | False | True |
| R1 | True | False | False | True |
| R2 | True | False | False | True |
| R3 | True | False | False | True |
| R4 | True | False | False | True |
| R5 | True | False | False | True |

Any periodic checkpoint found: False. Any best.pt found: False. All methods only have last endpoint checkpoints in outputs: False.

## Figures

- figures/all_methods_overview.png
- figures/core_methods.png
- figures/suspicious_groups.png
