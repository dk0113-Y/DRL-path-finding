from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.q_value_agent import ACTION_DIM, ExplorationQConfig, ExplorationQNetwork  # noqa: E402
from encoders.advantage_encoder import AdvantageEncoderConfig  # noqa: E402
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


ZEROED_CHANNELS = ["visit_count_log_norm", "recent_trajectory_decay"]


class DummyCumMap:
    def __init__(self) -> None:
        self.local_shape = (5, 5)
        self.map = np.full((7, 7), EMPTY, dtype=np.int8)
        self.map[2, 3] = OBSTACLE
        self.visit_count = np.ones((7, 7), dtype=np.int32)
        self.visit_count[3, 3] = 4
        self.visit_count[3, 2] = 3

    def world_to_array(self, world_rc: tuple[int, int]) -> tuple[int, int]:
        return int(world_rc[0]), int(world_rc[1])

    def get_frontier_u8(self, refresh: bool = False) -> np.ndarray:
        raise AssertionError("F_key advantage canvas must not request a frontier raster")


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


def _model_parameter_count() -> int:
    net = ExplorationQNetwork(
        ExplorationQConfig(
            advantage_encoder=AdvantageEncoderConfig(
                canvas_in_channels=4,
                canvas_channels=FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
            )
        )
    )
    return int(sum(parameter.numel() for parameter in net.parameters()))


def _check_f_key_dry_run() -> dict[str, object]:
    payload = _run_json(
        [
            sys.executable,
            "experiments/final_method/run_a_new_no_behavior_memory_ablation.py",
            "--run-stage",
            "formal",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    train_config = payload["train_config"]
    _assert(payload["experiment_id"] == "Anew_F", "F_key experiment_id mismatch")
    _assert(payload["method_id"] == "Anew_F3_no_behavior_memory", "F_key method_id mismatch")
    _assert(payload["method_name"] == "no_behavior_memory", "F_key method_name mismatch")
    _assert(payload["ablation_group"] == "input_state", "F_key ablation_group mismatch")
    _assert(payload["ablation_name"] == "no_behavior_memory", "F_key ablation_name mismatch")
    _assert(payload["channel_ablation"] == "no_behavior_memory", "F_key channel_ablation mismatch")
    _assert(payload["zeroed_advantage_channels"] == ZEROED_CHANNELS, "F_key zeroed channels mismatch")
    _assert(
        payload["advantage_canvas_schema"] == ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        "F_key advantage schema mismatch",
    )
    _assert(payload["advantage_canvas_channels"] == list(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS), "F_key channel list mismatch")
    _assert(int(payload["advantage_canvas_channel_count"]) == 4, "F_key channel count must stay 4")
    _assert(payload["frontier_raster_used"] is False, "F_key frontier_raster_used must be false")
    _assert(payload["value_tree_enabled"] is True, "F_key value_tree_enabled must be true")
    _assert(payload["value_tree_unchanged"] is True, "F_key value_tree_unchanged must be true")
    _assert(payload["value_replacement_strategy"] == "none", "F_key must not use value replacement")
    _assert(payload["dummy_value_tensors_for_interface"] is False, "F_key must not use D zero-value tensors")
    _assert(payload["dummy_value_mask_rule"] == "none", "F_key dummy value mask rule changed")
    _assert(payload["reward_override"] == {}, "F_key reward_override must be empty")
    _assert(payload["model_class"] == "ExplorationQNetwork", "F_key model class mismatch")
    _assert(int(payload["advantage_encoder.canvas_in_channels"]) == 4, "F_key encoder channels must stay 4")
    _assert(int(payload["model_parameter_count"]) == _model_parameter_count(), "F_key model parameter count mismatch")
    _assert(float(payload["reward_info_scale"]) == 3.1, "F_key reward_info_scale default changed")
    _assert(float(payload["reward_obstacle_weight"]) == 0.2, "F_key reward_obstacle_weight default changed")
    _assert(int(payload["learner_updates_per_iter"]) == 1, "F_key learner_updates_per_iter default changed")
    _assert(int(payload["min_replay_size"]) == 8000, "F_key min_replay_size default changed")
    _assert(float(payload["epsilon_end"]) == 0.04, "F_key epsilon_end default changed")
    _assert(int(payload["epsilon_decay_steps"]) == 240000, "F_key epsilon_decay_steps default changed")
    _assert(payload["train_side_only_tuning"] is True, "F_key train_side_only_tuning must default true")
    _assert(payload["occupancy_only_alias"] is True, "F_key should mark occupancy-only as an alias")
    _assert(payload["separate_occupancy_only_formal_row"] is False, "F_key must not create a separate occupancy-only row")
    _assert(payload["method_id"] != "Anew_F4_occupancy_only", "Anew_F4 must not be produced as method_id")
    _assert("Anew_F4_occupancy_only" not in str(payload["run_name"]), "Anew_F4 must not be produced as run_name")
    _assert("frontier_block_area_map" not in payload["advantage_canvas_channels"], "F_key restored frontier raster channel")
    _assert("frontier_block_area_map" not in " ".join(payload["train_args"]), "F_key train args should not use frontier_block_area_map")
    _assert(train_config["channel_ablation"] == "no_behavior_memory", "F_key train_config channel_ablation mismatch")
    _assert(train_config["zeroed_advantage_channels"] == ZEROED_CHANNELS, "F_key train_config zeroed channels mismatch")
    _assert(train_config["value_tree_enabled"] is True, "F_key train_config value tree changed")
    _assert(train_config["reward_override"] == {}, "F_key train_config reward override changed")
    return payload


def _check_a_new_contract_unchanged() -> None:
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
    _assert(train_config["channel_ablation"] == "none", "A_new channel_ablation changed")
    _assert(train_config["zeroed_advantage_channels"] == [], "A_new zeroed_advantage_channels changed")
    _assert(train_config["value_replacement_strategy"] == "none", "A_new replacement strategy changed")
    _assert(train_config["no_value_tree"] is False, "A_new no_value_tree changed")
    _assert(train_config["train_side_only_tuning"] is True, "A_new train_side_only_tuning changed")
    _assert("frontier_block_area_map" not in payload["advantage_canvas_channels"], "A_new restored frontier raster channel")


def _check_tensor_zeroing() -> torch.Tensor:
    cum_map = DummyCumMap()
    snapshot = _snapshot()
    recent_trajectory = [(3, 1), (3, 2), (3, 3)]
    base_builder = AdvantageStateBuilder(
        AdvantageStateConfig(
            advantage_canvas_schema=ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        )
    )
    f_key_builder = AdvantageStateBuilder(
        AdvantageStateConfig(
            advantage_canvas_schema=ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
            zeroed_advantage_channels=tuple(ZEROED_CHANNELS),
        )
    )
    base_canvas, _ = base_builder.build(
        cum_map,
        (3, 3),
        snapshot,
        recent_trajectory_positions=recent_trajectory,
    )
    f_key_canvas, meta = f_key_builder.build(
        cum_map,
        (3, 3),
        snapshot,
        recent_trajectory_positions=recent_trajectory,
    )
    _assert(f_key_canvas.shape == (4, 5, 5), f"F_key canvas shape mismatch: {f_key_canvas.shape}")
    _assert(float(meta["advantage_canvas_channel_count"]) == 4.0, "F_key canvas meta channel count mismatch")
    _assert(float(meta["zeroed_advantage_channel_count"]) == 2.0, "F_key zeroed channel meta mismatch")
    _assert(np.array_equal(f_key_canvas[0], base_canvas[0]), "free channel changed under F_key")
    _assert(np.array_equal(f_key_canvas[1], base_canvas[1]), "obstacle channel changed under F_key")
    _assert(float(base_canvas[0].sum()) > 0.0, "free channel should contain nonzero values")
    _assert(float(base_canvas[1].sum()) > 0.0, "obstacle channel should contain nonzero values")
    _assert(float(base_canvas[2].sum()) > 0.0, "base visit_count_log_norm should be nonzero in fixture")
    _assert(float(base_canvas[3].sum()) > 0.0, "base recent_trajectory_decay should be nonzero in fixture")
    _assert(float(f_key_canvas[2].sum()) == 0.0, "F_key visit_count_log_norm must be all zero")
    _assert(float(f_key_canvas[3].sum()) == 0.0, "F_key recent_trajectory_decay must be all zero")
    return torch.from_numpy(f_key_canvas).unsqueeze(0).repeat(2, 1, 1, 1)


def _check_forward_with_non_dummy_value_tree(advantage_canvas: torch.Tensor) -> None:
    value_builder = ValueStateBuilder()
    block, entry, block_mask, entry_mask, meta = value_builder.build(_snapshot())
    _assert(bool(block_mask.any()), "F_key value tree should contain non-dummy block masks")
    _assert(bool(entry_mask.any()), "F_key value tree should contain non-dummy entry masks")
    _assert(float(meta["value_packed_block_count"]) >= 1.0, "F_key value metadata should report packed blocks")
    _assert(float(meta["value_packed_entry_count"]) >= 1.0, "F_key value metadata should report packed entries")

    batch = int(advantage_canvas.shape[0])
    net = ExplorationQNetwork()
    net.eval()
    value_block_features = torch.from_numpy(block).unsqueeze(0).repeat(batch, 1, 1)
    value_entry_features = torch.from_numpy(entry).unsqueeze(0).repeat(batch, 1, 1, 1)
    value_block_mask = torch.from_numpy(block_mask).unsqueeze(0).repeat(batch, 1)
    value_entry_mask = torch.from_numpy(entry_mask).unsqueeze(0).repeat(batch, 1, 1)
    with torch.no_grad():
        q_values = net(
            advantage_canvas.to(dtype=torch.float32),
            value_block_features,
            value_entry_features,
            value_block_mask,
            value_entry_mask,
            return_aux=False,
        )
    _assert(q_values.shape == (batch, ACTION_DIM), f"Q shape mismatch: {tuple(q_values.shape)}")
    _assert(torch.isfinite(q_values).all().item(), "Q values must be finite")


def main() -> int:
    _check_f_key_dry_run()
    _check_a_new_contract_unchanged()
    advantage_canvas = _check_tensor_zeroing()
    _check_forward_with_non_dummy_value_tree(advantage_canvas)
    print("A_new no-behavior-memory ablation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
