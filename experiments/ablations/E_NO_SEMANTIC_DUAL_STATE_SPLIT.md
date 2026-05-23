# E No Semantic Dual-State Split

`E_ablation_no_semantic_dual_state_split` is a structural ablation for the full method decision head.

## Purpose

The full method encodes two semantic states and fuses them in `SemanticDuelingHead`:

- `value_state` from `ValueTreeEncoder`
- `advantage_state` from `AdvantageCanvasEncoder`

E keeps both information sources but removes the explicit dual-state dueling decision structure. It tests whether that separation and dueling fusion are useful, not whether value-tree information is useful.

## Model

`NoSemanticDualStateSplitQNetwork`:

- consumes the same forward inputs as `ExplorationQNetwork`
- calls `AdvantageCanvasEncoder`
- calls `ValueTreeEncoder`
- projects both encoder outputs into a fused per-action latent
- predicts one Q value per action through a single action-value head
- does not call `SemanticDuelingHead`

Aux markers:

- `no_semantic_dual_state_split = 1`
- `semantic_dual_state_split_used = 0`
- `value_tree_used_by_model = 1`

## Evidence Rule

E must be retrained from its own run. A, D, F, or R checkpoints cannot be reused as final E performance evidence.

Smoke and pilot runs only validate execution. They do not enter paper Results.
