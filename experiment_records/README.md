# Experiment Records

`main` only reserves tracked record space for the active A_new final method and
Anew_R1-Anew_R5 reward ablations.

- Active lightweight records belong under `experiment_records/final_method/`.
- Do not commit raw smoke outputs, raw run directories, checkpoint bodies, or
  large logs.
- Do not commit `outputs/`, `checkpoint_store/`, `checkpoints/`, `.pt`, `.pth`, or
  `.ckpt` files.

Legacy A/F1/F6/F7/ABCDEFR records, older full-method reference records, baseline
records, and final-probe matrices were archived before cleanup at:

- branch: `legacy/pre-a-new-cleanup`
- tag: `legacy-pre-a-new-cleanup-20260525`
