# DRL-path-finding

这是一个基于 PyTorch 的二维栅格地图自主探索强化学习工程。当前 `main`
已经清理为 A_new final 4-channel 主线：advantage 分支只接收局部 4 通道
canvas，value 分支继续接收由 `SharedSemanticSnapshot` 构造的 structured
frontier-block value tree。

## 当前主线

`A_new` 是新的 full_method_main：

- `method_id = A_new`
- `method_name = final_4ch_no_frontier_raster`
- `advantage_canvas_schema = final_4ch_no_frontier_raster`
- `advantage_canvas_channels = ["free", "obstacle", "visit_count_log_norm", "recent_trajectory_decay"]`
- `advantage_canvas_channel_count = 4`
- `frontier_raster_used = false`
- `value_tree_enabled = true`
- `value_branch_source = SharedSemanticSnapshot`
- `value_branch_representation = structured_frontier_block_value_tree`
- `model_class = ExplorationQNetwork`

Advantage canvas 不再包含 frontier raster。frontier、unknown block 和 frontier
cluster 语义仍保留在 shared semantic layer 与 value tree 中，用于 value branch。

旧 A/F1/F6/F7/ABCDEFR 实验入口、旧 frontier-raster diagnostics 和旧结果记录已从
active main 移除，并在清理前归档到：

- branch: `legacy/pre-a-new-cleanup`
- tag: `legacy-pre-a-new-cleanup-20260525`

## 代码结构

- `train_q_agent.py`: 主训练入口，负责配置、系统组装、训练循环、checkpoint
  selection、final probe 和 formal artifacts。
- `agents/q_value_agent.py`: `ExplorationQNetwork` 与 `StateTensorAdapter`。
- `env/advantage_state_builder.py`: A_new final 4-channel advantage canvas。
- `env/shared_semantic_layer.py`: shared semantic snapshot，包括 frontier / unknown
  block / cluster 构造逻辑。
- `env/value_state_builder.py`: structured frontier-block value tree。
- `experiments/final_method/`: A_new final method 与 Anew_R1-Anew_R5 reward
  ablation launchers。

## 运行方式

A_new smoke dry-run：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu -DryRun
```

A_new smoke：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage smoke -Device cpu
```

A_new formal：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_final_4ch.ps1 -RunStage formal -Device cuda
```

Anew_R1-Anew_R5 reward ablation dry-run：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_reward_ablations.ps1 -RunStage smoke -Device cpu -DryRun
```

Anew_R1-Anew_R5 formal：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_a_new_reward_ablations.ps1 -RunStage formal -Device cuda -NoCopyCheckpoints
```

`Anew_R1` through `Anew_R5` all bind to the A_new final 4-channel schema. Their
reward overrides are:

- `Anew_R1`: `reward_step_penalty = 0.0`
- `Anew_R2`: `reward_revisit_penalty = 0.0`
- `Anew_R3`: `reward_turn_penalty_scale = 0.0`
- `Anew_R4`: `reward_timeout_penalty = 0.0`
- `Anew_R5`: all four efficiency penalties above set to `0.0`

## Repository Hygiene

Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, or checkpoint files
such as `.pt`, `.pth`, and `.ckpt`. Smoke outputs are local artifacts only.
