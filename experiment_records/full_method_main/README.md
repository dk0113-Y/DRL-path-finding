# A Full Method 结果记录

本目录是 A Full method / 完整方法的正式结果记录位置。

后续应从训练设备复制 `logs/` 下的轻量 artifact 到本目录的 `logs/` 子目录。不要复制 checkpoints、模型权重、完整 outputs、replay buffer 或大体积调试文件。

建议在 `run_record.md` 中记录：

- base commit
- run name
- formal protocol
- final probe episodes
- `fixed_final_probe_seed_base`
- source workspace
- artifact 完整性核查结论
