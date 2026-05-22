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
- `ablations/` 下各目录对应 F/R 消融实验记录，只保存 curated logs，不保存完整 outputs 或 checkpoints。
- `templates/` 保存人工记录和 artifact 检查模板。
