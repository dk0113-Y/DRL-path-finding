from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.no_dual_state_split_q_network import (  # noqa: E402
    ACTION_DIM,
    NoDualStateSplitQConfig,
    NoDualStateSplitQNetwork,
    no_dual_state_split_model_parameter_count,
)
from env.advantage_state_builder import (  # noqa: E402
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
    AdvantageStateBuilder,
    AdvantageStateConfig,
)
from env.core_cummap import AnalysisBox  # noqa: E402
from env.grid_topology import EMPTY, OBSTACLE  # noqa: E402
from env.shared_semantic_layer import (  # noqa: E402
    FrontierCluster,
    SharedSemanticSnapshot,
    SparseMaskGeometry,
    SupportGeometry,
    UnknownBlock,
)
from env.value_state_builder import ValueStateBuilder  # noqa: E402


METHOD_ID = "Anew_E_no_dual_state_split"
METHOD_NAME = "no_dual_state_split_flat_value_injected_q"


class DummyCumMap:
    def __init__(self) -> None:
        self.local_shape = (5, 5)
        self.map = np.full((7, 7), EMPTY, dtype=np.int8)
        self.map[2, 3] = OBSTACLE
        self.map[3, 4] = OBSTACLE
        self.visit_count = np.ones((7, 7), dtype=np.int32)
        self.visit_count[3, 3] = 5
        self.visit_count[3, 2] = 3

    def world_to_array(self, world_rc: tuple[int, int]) -> tuple[int, int]:
        return int(world_rc[0]), int(world_rc[1])

    def get_frontier_u8(self, refresh: bool = False) -> np.ndarray:
        raise AssertionError("Anew_E no-dual-state-split must not request a frontier raster")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json_from_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start < 0:
        raise AssertionError(f"No JSON object found in output:\n{output}")
    return json.loads(output[start:])


def _run_json(command: list[str]) -> dict[str, object]:
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    return _load_json_from_output(result.stdout)


def _geometry_from_cells(cells: list[tuple[int, int]]) -> SparseMaskGeometry:
    if not cells:
        return SparseMaskGeometry.empty()
    rows = np.asarray([cell[0] for cell in cells], dtype=np.int32)
    cols = np.asarray([cell[1] for cell in cells], dtype=np.int32)
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max()) + 1
    c1 = int(cols.max()) + 1
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return SparseMaskGeometry(r0=r0, c0=c0, mask=mask, count=int(mask.sum()))


def _cluster(cells: list[tuple[int, int]], *, block_index: int) -> FrontierCluster:
    support = SupportGeometry(
        local_box_bounds=(0, 0, 0, 0),
        support_free_geometry=SparseMaskGeometry.empty(),
        support_obstacle_density=0.25,
    )
    anchor = cells[0] if cells else (0, 0)
    return FrontierCluster(
        frontier_index=block_index,
        block_index=block_index,
        frontier_geometry=_geometry_from_cells(cells),
        support_geometry=support,
        frontier_anchor_rc=(int(anchor[0]), int(anchor[1])),
        delta_r=1.0,
        delta_c=-1.0,
        entry_width=float(max(1, len(cells))),
    )


def _snapshot() -> SharedSemanticSnapshot:
    analysis_box = AnalysisBox(
        r0=0,
        r1=7,
        c0=0,
        c1=7,
        margin=0,
        known_r0=0,
        known_r1=7,
        known_c0=0,
        known_c1=7,
    )
    block = UnknownBlock(
        block_index=0,
        unknown_geometry=_geometry_from_cells([(1, 1), (1, 2), (2, 1)]),
        frontier_clusters=(_cluster([(2, 3), (3, 4)], block_index=0),),
        block_area=3,
        frontier_cluster_count=1,
    )
    return SharedSemanticSnapshot(
        analysis_box=analysis_box,
        accessible_blocks=(block,),
        total_accessible_unknown_area=3,
    )


def _check_e_dry_run() -> dict[str, object]:
    payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_no_dual_state_split_ablation.py",
            "--run-stage",
            "formal",
            "--device",
            "cuda",
            "--dry-run",
        ]
    )
    train_config = payload["train_config"]
    _assert(payload["experiment_id"] == "Anew_E", "E experiment_id mismatch")
    _assert(payload["method_id"] == METHOD_ID, "E method_id mismatch")
    _assert(payload["method_name"] == METHOD_NAME, "E method_name mismatch")
    _assert(payload["ablation_group"] == "structural", "E ablation_group mismatch")
    _assert(payload["ablation_id"] == "E", "E ablation_id mismatch")
    _assert(payload["ablation_name"] == "no_dual_state_split", "E ablation_name mismatch")
    _assert(payload["model_class"] == "NoDualStateSplitQNetwork", "E model class mismatch")
    _assert(int(payload["model_parameter_count"]) == no_dual_state_split_model_parameter_count(), "E parameter count mismatch")
    _assert(payload["dual_state_split_enabled"] is False, "E must disable dual state split")
    _assert(payload["explicit_advantage_value_split"] is False, "E must disable explicit advantage/value split")
    _assert(payload["semantic_dueling_head_used"] is False, "E must not use SemanticDuelingHead")
    _assert(payload["no_semantic_dueling_head"] is True, "E no_semantic_dueling_head flag mismatch")
    _assert(payload["value_tree_information_used"] is True, "E must keep value tree information")
    _assert(payload["value_tree_enabled"] is True, "E value_tree_enabled must stay true")
    _assert(payload["value_tree_branch_separate"] is False, "E must not keep a separate value branch")
    _assert(payload["value_tree_summary_injected"] is True, "E must inject flattened value summary")
    _assert(payload["value_replacement_strategy"] == "none", "E must not replace value tree tensors")
    _assert(payload["value_tensors_used_by_model"] is True, "E model must consume value tensors")
    _assert(payload["dummy_value_tensors_for_interface"] is False, "E must not use D dummy value tensors")
    _assert(payload["dummy_value_mask_rule"] == "none", "E dummy value mask rule changed")
    _assert(payload["behavior_memory_channels_used"] is True, "E must keep behavior-memory channels")
    _assert(payload["zeroed_advantage_channels"] == [], "E must not zero advantage channels")
    _assert(
        payload["advantage_canvas_schema"] == ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        "E advantage schema mismatch",
    )
    _assert(payload["advantage_canvas_channels"] == list(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS), "E channel list mismatch")
    _assert(int(payload["advantage_canvas_channel_count"]) == 4, "E channel count must be 4")
    _assert(payload["frontier_raster_used"] is False, "E frontier_raster_used must be false")
    _assert(payload["reward_override"] == {}, "E reward_override must be empty")
    _assert(float(payload["reward_info_scale"]) == 3.1, "E reward_info_scale default changed")
    _assert(float(payload["reward_obstacle_weight"]) == 0.2, "E reward_obstacle_weight default changed")
    _assert(float(payload["reward_step_penalty"]) == 0.02, "E reward_step_penalty default changed")
    _assert(float(payload["reward_terminal_bonus"]) == 20.0, "E reward_terminal_bonus default changed")
    _assert(float(payload["reward_revisit_penalty"]) == 0.1, "E reward_revisit_penalty default changed")
    _assert(float(payload["reward_turn_penalty_scale"]) == 0.05, "E reward_turn_penalty_scale default changed")
    _assert(float(payload["reward_timeout_penalty"]) == 8.0, "E reward_timeout_penalty default changed")
    _assert(int(payload["learner_updates_per_iter"]) == 1, "E learner_updates_per_iter default changed")
    _assert(int(payload["min_replay_size"]) == 8000, "E min_replay_size default changed")
    _assert(float(payload["epsilon_end"]) == 0.04, "E epsilon_end default changed")
    _assert(int(payload["epsilon_decay_steps"]) == 240000, "E epsilon_decay_steps default changed")
    _assert(payload["train_side_only_tuning"] is True, "E train_side_only_tuning must default true")
    _assert(train_config["model_class"] == "NoDualStateSplitQNetwork", "E train_config model class mismatch")
    _assert(train_config["dual_state_split_enabled"] is False, "E train_config dual split flag mismatch")
    _assert(train_config["semantic_dueling_head_used"] is False, "E train_config semantic head flag mismatch")
    _assert(train_config["value_tree_information_used"] is True, "E train_config value tree flag mismatch")
    _assert(train_config["reward_override"] == {}, "E train_config reward_override changed")
    dumped = json.dumps(payload, ensure_ascii=False)
    _assert("frontier_block_area_map" not in dumped, "E must not restore frontier_block_area_map")
    _assert("SemanticDuelingHead" not in dumped, "E dry-run must not mention SemanticDuelingHead")
    _assert("Anew_D_no_value_tree" not in dumped, "E must not degrade into D")
    _assert("Anew_F3_no_behavior_memory" not in dumped, "E must not degrade into F")
    _assert("Anew_C_local_state_ddqn" not in dumped, "E must not degrade into C")
    return payload


def _check_no_dual_model_source() -> None:
    source = inspect.getsource(NoDualStateSplitQNetwork)
    config_source = inspect.getsource(NoDualStateSplitQConfig)
    combined = "\n".join((source, config_source))
    forbidden = (
        "SemanticDuelingHead",
        "ValueTreeEncoder",
        "frontier_block_area_map",
        "FINAL_5CH",
    )
    for token in forbidden:
        _assert(token not in combined, f"E model source contains forbidden token: {token}")
    cfg = NoDualStateSplitQConfig()
    _assert(tuple(cfg.canvas_channels) == FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS, "E config channel order mismatch")
    _assert(int(cfg.canvas_in_channels) == 4, "E config must use 4 input channels")


def _check_forward_with_real_value_tree() -> None:
    torch.manual_seed(11)
    cum_map = DummyCumMap()
    snapshot = _snapshot()
    advantage_builder = AdvantageStateBuilder(
        AdvantageStateConfig(
            advantage_canvas_schema=ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        )
    )
    canvas, meta = advantage_builder.build(
        cum_map,
        (3, 3),
        snapshot,
        recent_trajectory_positions=[(3, 1), (3, 2), (3, 3)],
    )
    _assert(canvas.shape == (4, 5, 5), f"E canvas shape mismatch: {canvas.shape}")
    _assert(meta["frontier_raster_used"] is False, "E canvas must not use frontier raster")
    _assert(float(canvas[2].sum()) > 0.0, "E visit_count_log_norm channel must remain populated")
    _assert(float(canvas[3].sum()) > 0.0, "E recent_trajectory_decay channel must remain populated")

    value_builder = ValueStateBuilder()
    block, entry, block_mask, entry_mask, value_meta = value_builder.build(snapshot)
    _assert(bool(block_mask.any()), "E value block mask should contain real value-tree entries")
    _assert(bool(entry_mask.any()), "E value entry mask should contain real value-tree entries")
    _assert(float(value_meta["value_packed_block_count"]) >= 1.0, "E value metadata should report packed blocks")
    _assert(float(value_meta["value_packed_entry_count"]) >= 1.0, "E value metadata should report packed entries")

    batch = 2
    advantage_canvas = torch.from_numpy(canvas).unsqueeze(0).repeat(batch, 1, 1, 1).to(dtype=torch.float32)
    value_block_features = torch.from_numpy(block).unsqueeze(0).repeat(batch, 1, 1).to(dtype=torch.float32)
    value_entry_features = torch.from_numpy(entry).unsqueeze(0).repeat(batch, 1, 1, 1).to(dtype=torch.float32)
    value_block_mask = torch.from_numpy(block_mask).unsqueeze(0).repeat(batch, 1)
    value_entry_mask = torch.from_numpy(entry_mask).unsqueeze(0).repeat(batch, 1, 1)

    net = NoDualStateSplitQNetwork()
    net.eval()
    with torch.no_grad():
        q_values, aux = net(
            advantage_canvas,
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=True,
        )
        q_zero_value = net(
            advantage_canvas,
            torch.zeros_like(value_block_features),
            torch.zeros_like(value_entry_features),
            value_block_mask,
            value_entry_mask,
            return_aux=False,
        )
    _assert(q_values.shape == (batch, ACTION_DIM), f"E Q shape mismatch: {tuple(q_values.shape)}")
    _assert(torch.isfinite(q_values).all().item(), "E Q values must be finite")
    _assert(torch.isfinite(q_zero_value).all().item(), "E zero-value Q values must be finite")
    _assert(not torch.allclose(q_values, q_zero_value), "E Q output must depend on value-tree feature tensors")
    _assert("no_dual_value_summary_norm" in aux, "E aux missing value summary norm")
    _assert("no_dual_value_valid_block_count" in aux, "E aux missing value block count")
    _assert("no_dual_local_action_feature_norm" in aux, "E aux missing local action feature norm")
    _assert(torch.all(aux["no_dual_value_valid_block_count"] >= 1).item(), "E aux value block count should be positive")


def _check_existing_a_new_contract_unchanged() -> None:
    payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_final_method.py",
            "--run-stage",
            "formal",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    train_config = payload["train_config"]
    _assert(payload["method_id"] == "A_new", "A_new dry-run method_id changed")
    _assert(payload["method_name"] == "final_4ch_no_frontier_raster", "A_new dry-run method_name changed")
    _assert(int(payload["advantage_canvas_channel_count"]) == 4, "A_new channel count changed")
    _assert(payload["frontier_raster_used"] is False, "A_new frontier_raster_used changed")
    _assert(payload["value_tree_enabled"] is True, "A_new value_tree_enabled changed")
    _assert(payload["reward_override"] == {}, "A_new reward_override changed")
    _assert(train_config["model_class"] == "ExplorationQNetwork", "A_new model class changed")
    _assert(train_config["dual_state_split_enabled"] is True, "A_new dual-state split flag changed")
    _assert(train_config["semantic_dueling_head_used"] is True, "A_new semantic head flag changed")
    _assert(train_config["value_tree_information_used"] is True, "A_new value-tree flag changed")
    _assert(train_config["value_replacement_strategy"] == "none", "A_new replacement strategy changed")
    _assert(train_config["no_value_tree"] is False, "A_new no_value_tree changed")
    _assert("frontier_block_area_map" not in payload["advantage_canvas_channels"], "A_new restored frontier raster channel")


def main() -> int:
    _check_e_dry_run()
    _check_no_dual_model_source()
    _check_forward_with_real_value_tree()
    _check_existing_a_new_contract_unchanged()
    print("A_new no-dual-state-split ablation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
