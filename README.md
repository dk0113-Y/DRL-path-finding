# DRL-path-finding

Portfolio-oriented research code for deep reinforcement learning based
autonomous exploration in a 2D grid-world robot simulation. The active branch is
organized around the A_new final method: a Double-DQN exploration agent that
uses a local 4-channel action-advantage canvas together with a structured
frontier-block value tree.

This repository is intended to support internship applications in deep
reinforcement learning, mobile robot autonomous exploration, path planning, and
simulation-based evaluation. It is not a product demo, a ROS deployment, or a
physical-robot validation package.

## Contents

- [Project Overview](#project-overview)
- [Research Relation](#research-relation)
- [Problem Formulation](#problem-formulation)
- [Method Highlights](#method-highlights)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Core Modules](#core-modules)
- [Training and Evaluation Workflow](#training-and-evaluation-workflow)
- [Quick Start](#quick-start)
- [Common Commands](#common-commands)
- [Experiments and Metrics](#experiments-and-metrics)
- [Current Status and Limitations](#current-status-and-limitations)
- [Internship Skill Mapping](#internship-skill-mapping)
- [Advanced Notes](#advanced-notes)
- [Repository Hygiene](#repository-hygiene)

## Project Overview

The project studies the following robot exploration problem:

- A mobile robot starts in an unknown 2D occupancy-grid map.
- At each step it receives a local radar-style observation and updates a
  cumulative belief map.
- The policy selects one of 8 grid actions: N, NE, E, SE, S, SW, W, NW.
- The episode succeeds when effective map coverage reaches the configured
  threshold, currently `coverage_stop_threshold = 0.95` for the active formal
  contract.
- Training and evaluation track reward, coverage, success rate, episode length,
  repeat visits, timeout rate, information gain, frontier/value-tree diagnostics,
  and reproducibility metadata.

The current maintained mainline is `A_new` /
`final_4ch_no_frontier_raster`. It keeps frontier and unknown-region semantics
in a structured value branch, while the local action branch uses only occupancy
and behavior-memory channels.

## Research Relation

This repository is the code and experiment workspace behind the author's
research direction on frontier-guided DRL end-to-end autonomous exploration. The
author-provided paper relation is:

- Paper: `基于前沿引导的 DRL 端到端自主探索算法`
- Role: first author
- Indexing/status: EI paper, published / electronically available according to
  the author-provided project background

Important evidence boundaries:

- The paper and repository are related, but this README does not restate the
  full paper or claim additional unpublished results.
- `A_new` is the current code mainline in this repository.
- `experiment_records/final_method/unified_final_probe/` contains the current
  paper-facing held-out comparison records that are committed as lightweight
  CSV/JSON artifacts.
- Smoke and pilot runs are local execution checks only; they should not be
  treated as paper Results.
- Historical/internal matrices such as `Anew_B`, `Anew_C`, `Anew_D`, `Anew_E`,
  `Anew_F3`, and `Anew_R*` are useful for explaining baselines and ablations,
  but each row has its own evidence boundary.

## Problem Formulation

The environment is a random rectangular-obstacle occupancy grid implemented in
`env/`. The default formal A_new configuration in `train_q_agent.py` uses:

- Grid size: `rows = 40`, `cols = 60`
- Obstacle ratio: `0.20`
- Local sensor radius: `scan_radius = 10`
- Max episode length: `600`
- Coverage stop threshold: `0.95`
- Training budget: `total_env_steps = 650000`

The agent state is built from cumulative belief, not from an oracle full-map
decision input. The simulator still uses the true grid for environment stepping,
sensing, termination, and metric calculation.

Reward terms include:

- information gain from newly observed free and obstacle cells
- step penalty
- recent-revisit penalty
- turn penalty
- timeout penalty
- terminal success bonus

The active frozen V1 formal defaults include `reward_info_scale = 3.1`,
`reward_obstacle_weight = 0.2`, `reward_revisit_penalty = 0.12`,
`reward_turn_penalty_scale = 0.06`, `reward_timeout_penalty = 10.0`,
`epsilon_end = 0.03`, and `epsilon_decay_steps = 300000`.

## Method Highlights

### A_new Final Method

`A_new` is the active full-method mainline:

- `method_id = A_new`
- `method_name = final_4ch_no_frontier_raster`
- `model_class = ExplorationQNetwork`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- `value_tree_enabled = true`
- `value_branch_source = SharedSemanticSnapshot`
- `value_branch_representation = structured_frontier_block_value_tree`

### Local Advantage Canvas

`env/advantage_state_builder.py` builds the final 4-channel local canvas:

1. `free`
2. `obstacle`
3. `visit_count_log_norm`
4. `recent_trajectory_decay`

The current advantage branch intentionally does not use a frontier raster
channel.

### Frontier-Guided Value Tree

`env/shared_semantic_layer.py` extracts frontier/unknown-region semantics:

- `UnknownBlock`
- `FrontierCluster`
- `SupportGeometry`

`env/value_state_builder.py` packs these semantics into structured value-tree
tensors. Block features include `block_area_ratio` and
`frontier_cluster_count`; entry features include relative direction, entry
width, and support-obstacle-density cues.

### Dueling Q Network

`agents/q_value_agent.py` defines `ExplorationQNetwork`:

- `AdvantageCanvasEncoder` encodes local action-conditioned features.
- `ValueTreeEncoder` encodes structured frontier-block context.
- `SemanticDuelingHead` combines `V(s)` and `A(s,a)` into Q values.

`training/learner.py` implements a Double-DQN learner with hard target-network
sync, replay sampling, n-step targets through the replay pipeline, smooth L1 TD
loss, target masking, and optional CUDA AMP paths.

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib / PIL for plotting and artifact export tools
- PowerShell launchers for Windows experiment orchestration
- CSV/JSON experiment records for lightweight reproducibility artifacts

There is no package installer or dependency lock file in the current repository.
Install the runtime dependencies explicitly in your own Python environment.

## Repository Structure

```text
.
|-- README.md
|-- train_q_agent.py
|-- agents/
|   |-- q_value_agent.py
|   |-- local_state_q_network.py
|   `-- no_dual_state_split_q_network.py
|-- encoders/
|   |-- advantage_encoder.py
|   |-- value_encoder.py
|   |-- local_encoder.py
|   `-- global_encoder.py
|-- env/
|   |-- block_random_g.py
|   |-- agent_version.py
|   |-- core_cummap.py
|   |-- core_radar.py
|   |-- advantage_state_builder.py
|   |-- shared_semantic_layer.py
|   `-- value_state_builder.py
|-- heads/
|   `-- semantic_dueling_head.py
|-- training/
|   |-- collector.py
|   |-- learner.py
|   |-- replay_buffer.py
|   |-- evaluator.py
|   |-- checkpointing.py
|   |-- formal_artifacts.py
|   `-- plotting.py
|-- experiments/
|   `-- final_method/
|-- experiment_records/
|-- scripts/
|-- docs/
|-- demos/
`-- tools/
```

Notes:

- Active baselines and ablations live under `experiments/final_method/`.
- The local worktree may contain ignored `outputs/`, `__pycache__/`, and legacy
  cache folders. They are not part of the tracked code surface.
- Checkpoints and raw run directories are intentionally excluded from version
  control.

## Core Modules

| Area | Files | Role |
|---|---|---|
| Training entrypoint | `train_q_agent.py` | Parses configuration, builds components, runs training, checkpoint selection, and artifact writing. |
| Main agent | `agents/q_value_agent.py` | Defines `ExplorationQNetwork`, `StateTensorAdapter`, action masking, and semantic state tensors. |
| Baseline agent | `agents/local_state_q_network.py` | Defines the local 3-channel DDQN baseline model used by `Anew_C`. |
| Structural ablation model | `agents/no_dual_state_split_q_network.py` | Defines the no-dual-state-split ablation used by `Anew_E`. |
| Advantage encoding | `encoders/advantage_encoder.py` | Encodes the 4-channel local canvas into action-conditioned advantage states. |
| Value encoding | `encoders/value_encoder.py` | Encodes frontier-block value-tree tensors into a state-value representation. |
| Decision head | `heads/semantic_dueling_head.py` | Combines value and advantage streams into Q values. |
| Map simulation | `env/block_random_g.py`, `env/agent_version.py`, `env/core_radar.py` | Generates obstacle maps and local radar observations. |
| Belief and frontier state | `env/core_cummap.py`, `env/shared_semantic_layer.py` | Maintains belief, coverage, frontier cache, unknown blocks, and frontier clusters. |
| RL loop | `training/collector.py`, `training/learner.py`, `training/replay_buffer.py`, `training/evaluator.py` | Handles rollouts, replay, Double-DQN updates, and greedy evaluation. |
| Experiment launchers | `experiments/final_method/`, `scripts/` | Provide A_new, baselines, ablations, batch launchers, and final probes. |
| Records | `experiment_records/` | Stores curated lightweight experiment records and summary CSV/JSON files. |
| Demos/tools | `demos/`, `tools/` | Interactive semantic visualization, checks, plotting, export, and artifact utility scripts. |

## Training and Evaluation Workflow

Typical workflow:

1. Run a dry-run command to print the resolved experiment contract.
2. Run a CPU smoke check for fast local validation.
3. Run formal training on CUDA when a GPU training environment is available.
4. Archive only lightweight logs/CSV/JSON records.
5. Keep checkpoints in local `checkpoint_store/` or `outputs/`; do not commit
   `.pt`, `.pth`, or `.ckpt` files.
6. Run unified final probe only after the required checkpoints are present.

The active training loop supports:

- Double-DQN target computation
- target-network hard sync
- replay buffer and prioritized priority updates
- n-step transition builder
- epsilon decay
- fixed-seed final probes
- post-hoc formal checkpoint selection records

## Quick Start

From the repository root:

```powershell
python --version
python agents\q_value_agent.py
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu -DryRun
```

The first command checks your Python environment. The second runs the semantic
network smoke test embedded in `agents/q_value_agent.py`. The third prints the
A_new smoke contract without starting training.

If imports fail, install the runtime libraries used by the codebase, especially
PyTorch, NumPy, Matplotlib, and Pillow. This repository currently does not
provide a `requirements.txt`.

## Common Commands

A_new dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu -DryRun
```

A_new smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu
```

A_new formal training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage formal -Device cuda
```

Minimum-closure batch dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_minimum_closure_batch.ps1 -RunStage formal -Device cuda -DryRun
```

Unified final probe:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_unified_final_probe.ps1 -Device cuda
```

Environment-shift probe dry-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_environment_shift_final_probe.ps1 -Device cuda -DryRun
```

## Experiments and Metrics

The current held-out comparison record is:

```text
experiment_records/final_method/unified_final_probe/unified_final_probe_20260527_103303/
```

It uses `episodes = 100` and the fixed seed block
`20261323` through `20261422`. The committed summary reports:

| Label | Method | Coverage | Success rate | Episode length | Timeout rate |
|---|---|---:|---:|---:|---:|
| B | `Anew_B_classical_frontier_greedy` | 0.642881 | 0.21 | 515.59 | 0.79 |
| A | `AN` / `final_4ch_no_frontier_raster` | 0.937350 | 0.92 | 291.98 | 0.08 |
| C | `Anew_C_local_state_ddqn` | 0.496706 | 0.00 | 600.00 | 1.00 |
| D | `Anew_D_no_value_tree` | 0.931999 | 0.82 | 323.52 | 0.18 |
| E | `Anew_E_no_dual_state_split` | 0.925461 | 0.81 | 362.62 | 0.19 |
| F_key | `Anew_F3_no_behavior_memory` | 0.587208 | 0.00 | 600.00 | 1.00 |
| R_key | `Anew_R5` / `no_efficiency_penalties` | 0.927040 | 0.86 | 333.37 | 0.14 |

Interpretation boundary:

- These are repository experiment records, not newly rerun results.
- Some summary rows reference checkpoint paths from the original training
  machine; the checkpoint binaries are intentionally not committed.
- The table supports a portfolio narrative about controlled baselines and
  ablations, but it should not be overstated as physical-robot validation or
  broad DRL superiority.

## Current Status and Limitations

Current status:

- Active mainline is A_new final 4-channel no-frontier-raster.
- B classical frontier greedy, C local-state DDQN, D no-value-tree, E
  no-dual-state-split, F_key no-behavior-memory, and R_key reward-ablation rows
  have maintained launchers or records under `experiments/final_method/` and
  `experiment_records/`.
- Lightweight experiment records are committed.
- Raw outputs and checkpoint files are ignored by `.gitignore`.

Limitations:

- This is a 2D grid-world simulation repository, not a ROS stack.
- No physical robot validation is provided here.
- No trained checkpoint binaries are committed, so checkpoint-dependent final
  probes require the user's local `checkpoint_store/`.
- `outputs/` may exist locally with smoke artifacts; these are ignored local
  files and should not be pushed.
- Some historical records contain absolute paths from the original training
  machine. They are useful for provenance but should be sanitized if a fully
  public release requires path privacy.

## Internship Skill Mapping

| Internship skill area | Repository evidence |
|---|---|
| Deep Reinforcement Learning | Double-DQN learner, replay buffer, n-step transitions, epsilon schedule, and training loop. |
| Value-based RL / DQN / Double-DQN | `DDQNLearner` computes online argmax actions and target-network Q values. |
| Autonomous exploration | Belief-map maintenance, frontier/unknown semantics, coverage metrics, and success metrics. |
| Path planning and frontier baseline | `Anew_B_classical_frontier_greedy` uses reachable frontiers and BFS over known-free cells. |
| Grid-world simulation | Random obstacle maps, local observation, and radar-style sensing are implemented in `env/`. |
| Reward design | Information gain, step, revisit, turn, timeout, and terminal reward terms are configurable. |
| Observation/state representation | A_new uses a 4-channel local canvas plus structured frontier-block value-tree tensors. |
| Ablation experiment design | D, E, F_key, and R_key isolate value-tree, state-split, behavior-memory, and reward effects. |
| Baseline comparison | B is a classical non-learning frontier baseline; C is a simpler learning baseline with local state only. |
| Training/evaluation pipeline | Launchers support smoke, pilot, formal training, batch runs, and final probes. |
| PyTorch engineering | Encoders, semantic dueling head, target networks, AMP-aware learner paths, and tensor adapters. |
| Reproducibility and records | Config snapshots, metric snapshots, train CSVs, summaries, and fixed-seed probe records. |

## Advanced Notes

### Active A_new Contract

Frozen V1 formal defaults are pinned to the AN_tuned_v1 last.pt-oriented
training contract used by the current final experiment records and unified final
probe:

- `reward_info_scale = 3.1`
- `reward_obstacle_weight = 0.2`
- `learner_updates_per_iter = 1`
- `min_replay_size = 8000`
- `total_env_steps = 650000`
- `epsilon_end = 0.03`
- `epsilon_decay_steps = 300000`
- `reward_revisit_penalty = 0.12`
- `reward_turn_penalty_scale = 0.06`
- `reward_timeout_penalty = 10.0`
- `train_side_only_tuning = true` for the minimum-closure comparison rows

### Internal Experiment Matrix

Supported or recorded A_new-aligned rows:

- `A_new`: final 4-channel no-frontier-raster method with structured value tree.
- `Anew_B`: classical frontier greedy, non-learning baseline, CPU benchmark.
- `Anew_C`: local-state DDQN learning baseline.
- `Anew_D`: no structured value tree.
- `Anew_E`: no dual-state split / flattened value-injected Q ablation.
- `Anew_F3`: no behavior-memory channels in the advantage canvas.
- `Anew_R1`: no step penalty.
- `Anew_R2`: no revisit penalty.
- `Anew_R3`: no turn penalty.
- `Anew_R4`: no timeout penalty.
- `Anew_R5`: no efficiency penalties, used as `R_key`.

Legacy A/F1/F6/F7/ABCDEFR launchers and frontier-raster diagnostics are not
active workflow entries on `main`. Historical cleanup state is preserved by the
tag:

```text
legacy-pre-a-new-cleanup-20260525
```

## Repository Hygiene

Do not commit:

- `outputs/`
- `checkpoint_store/`
- `checkpoints/`
- `.pt`, `.pth`, `.ckpt`
- replay buffers
- profiling/debug dumps
- generated cache files such as `__pycache__/`

Before publishing a public portfolio version, review committed experiment
records for absolute local paths and decide whether to sanitize them while
preserving reproducibility context.
