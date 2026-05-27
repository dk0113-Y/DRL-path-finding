# A_new Reward Ablation Run Record

- method_id: Anew_R5
- selector: R5
- name: no_efficiency_penalties
- baseline_method: A_new_final_4ch_no_frontier_raster
- run_stage: formal
- source run_dir: C:\Users\Dk\Desktop\SCI\New_A\outputs\Anew_R5_no_efficiency_penalties_formal_20260527_024235
- reward_override: {"reward_revisit_penalty": 0.0, "reward_step_penalty": 0.0, "reward_timeout_penalty": 0.0, "reward_turn_penalty_scale": 0.0}
- copied artifact list: metric_snapshot.json, config_snapshot.json, reproducibility_contract.json, artifact_index.json, training_summary.txt
- missing artifact list: final_probe.csv, final_probe_summary.json, posthoc_selection_summary.json, formal_selection_manifest.json
- checkpoint_source: C:\Users\Dk\Desktop\SCI\New_A\outputs\Anew_R5_no_efficiency_penalties_formal_20260527_024235\checkpoints\last.pt
- checkpoint_store_path: checkpoint_store\final_method\A_new_reward_ablations\Anew_R5.pt
- checkpoint_copied: true
- checkpoint_copy_reason: none

## Method Contract

- advantage_canvas_schema: final_4ch_no_frontier_raster
- frontier_raster_used: false
- value_tree_enabled: true
- model_class: ExplorationQNetwork
- advantage_encoder.canvas_in_channels: 4
