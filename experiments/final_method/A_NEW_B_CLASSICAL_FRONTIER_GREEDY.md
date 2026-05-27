# Anew_B Classical Frontier Greedy Baseline

`Anew_B_classical_frontier_greedy` is the A_new-aligned B group classical
frontier greedy baseline. It is a traditional non-learning exploration policy
for comparing A_new against a deterministic frontier-based exploration rule.

## Contract

- `experiment_id = Anew_B`
- `method_id = Anew_B_classical_frontier_greedy`
- `method_name = classical_frontier_greedy`
- `baseline_group = classical`
- `baseline_type = traditional_non_learning`
- `is_learning_baseline = false`
- `trainable_parameters = 0`
- `checkpoint_used = false`
- `no_ground_truth_map_for_decision = true`
- `reward_override = {}`
- `advantage_canvas_schema = not_applicable`
- `frontier_raster_used = false`
- `value_tree_enabled = not_applicable`

The baseline uses the current A_new environment, reward, seed, and metric
contract while restoring the legacy `classical_frontier_greedy_v1` policy
logic from `DRL_PF`. It does not train a model, load a checkpoint, or use
`ExplorationQNetwork`.

## Policy Rule

The policy chooses among valid actions using only belief-derived state:

1. Build frontier candidates from shared semantic frontier-cluster anchors, or
   fall back to the cumulative frontier cache.
2. Compute BFS cost from the current pose to reachable frontier anchors over
   known-free belief cells.
3. Select the lowest-cost reachable frontier target with legacy deterministic
   tie-breaks.
4. Move one valid action chosen by squared Euclidean distance to that target,
   then recent-trajectory revisit flag, visit count, and fixed `ACTIONS_8`
   order: N, NE, E, SE, S, SW, W, NW.
5. If no reachable frontier exists, choose the valid next action with the
   largest belief-only expected immediate information gain using radar
   line-of-sight over the current belief.

Policy decision inputs do not include the full map, map generator internals,
future sensor information, or shortest paths through unknown space. The runner
uses the simulator map only for environment stepping, sensor update,
termination, and metric computation.

## Artifacts

The baseline runner writes a baseline artifact package under `outputs/`:

- `logs/config_snapshot.json`
- `logs/baseline_manifest.json`
- `logs/baseline_policy_summary.json`
- `logs/benchmark_summary.json`
- `logs/metric_snapshot.json`
- `logs/reproducibility_contract.json`
- `logs/artifact_index.json`
- `logs/final_probe.csv`
- `logs/baseline_summary.txt`

`logs/final_probe.csv` is a non-learning baseline benchmark episode table, not a
neural model final probe.

Smoke and pilot runs are local checks only, not paper Results. B formal
benchmark artifacts can support a classical baseline comparison after artifact
review, but B cannot replace D/F/R internal ablations or explain neural
representation contributions.

## Launch

Dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage formal -Device cpu -DryRun
```

Smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage smoke -Device cpu
```

Formal benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_classical_frontier_baseline.ps1 -RunStage formal -Device cpu
```
