# D No Value Tree Structural Ablation

D is a structural ablation, not an F channel ablation, not an R reward ablation,
and not the B classical frontier baseline.

It preserves the full `ExplorationQNetwork` structure and input shapes, keeps
the 5-channel `advantage_canvas` unchanged, and replaces only the value branch
inputs with `value_replacement_strategy=zero_value_state`.

The zero-value state uses all-zero `value_block_features` and
`value_entry_features` with all-false `value_block_mask` and `value_entry_mask`.
No dummy node is used. The value encoder's masked softmax path handles this case
with zero attention weights, so the forward pass remains finite without leaking
real unknown block, frontier cluster, entry geometry, or area information.

Smoke check:

```powershell
python scripts\check_no_value_tree_ablation.py
python experiments\ablations\run_ablation_train.py --ablation-id D_ablation_no_value_tree --run-stage smoke -- --device cpu --output-root outputs
```

Formal template, to run later on the experiment machine:

```powershell
python experiments\ablations\run_ablation_train.py --ablation-id D_ablation_no_value_tree --run-stage formal -- --device cuda --rows 40 --cols 60 --obs-size 6 --scan-radius 10 --obstacle-ratio 0.20 --max-episode-steps 600 --coverage-stop-threshold 0.95 --trajectory-history-steps 10 --final-greedy-episodes 100 --fixed-final-probe-seed-base 20261323 --reward-info-scale 3.1 --reward-obstacle-weight 0.2 --reward-step-penalty 0.02 --reward-terminal-bonus 20.0 --reward-revisit-penalty 0.1 --reward-turn-penalty-scale 0.05 --reward-timeout-penalty 8.0
```

Smoke and pilot outputs stay under `outputs/` and are not paper results.
Checkpoints remain local under run output directories or `checkpoint_store/`,
which are ignored by Git.

