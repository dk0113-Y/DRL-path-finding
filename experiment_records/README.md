# 轻量实验结果记录

本目录用于保存论文实验的轻量结果记录，不是原始训练输出目录。这里应只放经过筛选的 evidence logs、记录模板和人工核查记录。

## 允许提交

- `logs/final_probe.csv`
- `logs/final_probe_summary.json`
- `logs/metric_snapshot.json`
- `logs/config_snapshot.json`
- `logs/reproducibility_contract.json`
- `logs/posthoc_selection_summary.json`
- `logs/formal_selection_manifest.json`
- `logs/artifact_index.json`
- `logs/ablation_manifest.json`，如果是消融实验
- `logs/baseline_manifest.json`，如果是 learning baseline
- `run_record.md`

## 禁止提交

- `checkpoints/`
- `*.pt`
- `*.pth`
- `*.ckpt`
- replay buffer
- raw `outputs/`
- profiling traces
- debug dumps

## 结果准入

smoke/pilot run 不进入论文 Results。只有 formal run 才能作为结果候选，并且需要先核查 artifact 完整性、seed policy、final probe protocol 和 run stage。

## 目录说明

- `full_method_main/` 是 A Full method / 完整方法的 reference run 记录位置。
- `ablations/` 下各目录对应结构、输入通道和 reward 消融实验记录，目录 slug 为 `<short_id>_ablation_<canonical_id>`，例如 `D_ablation_no_value_tree`、`E_ablation_no_semantic_dual_state_split`、`F1_ablation_no_frontier_channel` 和 `R5_ablation_no_efficiency_penalties`。
- `baselines/` 下各目录对应 learning baseline 记录，例如 `C_baseline_local_state_ddqn`。该目录只保存 curated logs 和 `run_record.md`，不保存 checkpoint 本体。
- `templates/` 保存人工记录和 artifact 检查模板。

outputs run_name、`experiment_records` 目录和 checkpoint_store 文件名应保持同一 slug。`experiment_records` 不保存 checkpoint 本体；checkpoint 本体保存在本地 `checkpoint_store/`，该目录被 Git 忽略，不应提交模型权重或 checkpoint。`run_record.md` 会记录 checkpoint source、checkpoint_store 路径和复制状态。
