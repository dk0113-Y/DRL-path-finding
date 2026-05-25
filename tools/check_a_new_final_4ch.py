from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.q_value_agent import ExplorationQConfig, ExplorationQNetwork  # noqa: E402
from encoders.advantage_encoder import AdvantageEncoderConfig  # noqa: E402
from env.advantage_state_builder import (  # noqa: E402
    ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
    FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
    AdvantageStateBuilder,
    AdvantageStateConfig,
)
from env.core_cummap import AnalysisBox  # noqa: E402
from env.grid_topology import EMPTY  # noqa: E402
from env.shared_semantic_layer import (  # noqa: E402
    FrontierCluster,
    SharedSemanticSnapshot,
    SparseMaskGeometry,
    SupportGeometry,
    UnknownBlock,
)
from env.value_state_builder import (  # noqa: E402
    VALUE_BLOCK_FEATURE_COUNT,
    VALUE_ENTRY_FEATURE_COUNT,
    ValueStateBuilder,
)


class DummyCumMap:
    def __init__(self) -> None:
        self.local_shape = (5, 5)
        self.map = np.full((7, 7), EMPTY, dtype=np.int8)
        self.visit_count = np.ones((7, 7), dtype=np.int32)
        self.visit_count[3, 3] = 4

    def world_to_array(self, world_rc: tuple[int, int]) -> tuple[int, int]:
        return int(world_rc[0]), int(world_rc[1])

    def get_frontier_u8(self, refresh: bool = False) -> np.ndarray:
        raise AssertionError("A_new final 4-channel canvas must not request a frontier raster")


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
        unknown_geometry=SparseMaskGeometry.empty(),
        frontier_clusters=(_cluster([(2, 3), (3, 4)], block_index=0),),
        block_area=5,
        frontier_cluster_count=1,
    )
    return SharedSemanticSnapshot(
        analysis_box=analysis_box,
        accessible_blocks=(block,),
        total_accessible_unknown_area=10,
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _first_conv_in_channels(model: ExplorationQNetwork) -> int:
    first = model.advantage_encoder.backbone[0]
    return int(first.in_channels)


def _load_json_from_output(output: str) -> dict[str, object]:
    start = output.find("{")
    if start < 0:
        raise AssertionError(f"No JSON object found in output:\n{output}")
    return json.loads(output[start:])


def _check_final_canvas() -> None:
    cum_map = DummyCumMap()
    snapshot = _snapshot()
    builder = AdvantageStateBuilder(
        AdvantageStateConfig(
            advantage_canvas_schema=ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER,
        )
    )

    canvas, meta = builder.build(
        cum_map,
        (3, 3),
        snapshot,
        recent_trajectory_positions=[(3, 1), (3, 2), (3, 3)],
    )
    _assert(canvas.shape == (4, 5, 5), f"expected final canvas shape (4,5,5), got {canvas.shape}")
    _assert(
        tuple(builder.config.advantage_canvas_channels) == FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
        "final channel order mismatch",
    )
    _assert("frontier_block_area_map" not in builder.config.advantage_canvas_channels, "frontier raster channel still present")
    _assert(meta.get("frontier_raster_used") is False, "frontier_raster_used must be false")
    _assert(float(meta.get("advantage_canvas_channel_count", 0.0)) == 4.0, "final channel count meta must be 4")


def _check_network_and_value_tree() -> None:
    net = ExplorationQNetwork(
        ExplorationQConfig(
            advantage_encoder=AdvantageEncoderConfig(
                canvas_in_channels=4,
                canvas_channels=FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS,
            )
        )
    )
    _assert(_first_conv_in_channels(net) == 4, "A_new first Conv2d must use 4 input channels")

    value_builder = ValueStateBuilder()
    block, entry, block_mask, entry_mask, meta = value_builder.build(_snapshot())
    _assert(block.shape[1] == VALUE_BLOCK_FEATURE_COUNT, "value block feature dim mismatch")
    _assert(entry.shape[2] == VALUE_ENTRY_FEATURE_COUNT, "value entry feature dim mismatch")
    _assert(bool(block_mask[0]), "value block mask should contain the dummy block")
    _assert(bool(entry_mask[0, 0]), "value entry mask should contain the dummy frontier entry")
    _assert(float(meta["value_packed_entry_count"]) >= 1.0, "value entry metadata was not built")


def _check_a_new_final_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/final_method/run_a_new_final_method.py",
            "--run-stage",
            "smoke",
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    payload = _load_json_from_output(result.stdout)
    _assert(payload["method_id"] == "A_new", "A_new dry-run method_id mismatch")
    _assert(payload["method_name"] == "final_4ch_no_frontier_raster", "A_new method_name mismatch")
    _assert(payload["advantage_canvas_schema"] == ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER, "schema mismatch")
    _assert(payload["advantage_canvas_channels"] == list(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS), "channel list mismatch")
    _assert(int(payload["advantage_canvas_channel_count"]) == 4, "A_new dry-run channel count must be 4")
    _assert(payload["frontier_raster_used"] is False, "A_new dry-run frontier_raster_used must be false")
    _assert(payload["value_tree_enabled"] is True, "value tree must stay enabled")
    _assert(payload["model_class"] == "ExplorationQNetwork", "model class mismatch")
    _assert(int(payload["advantage_encoder.canvas_in_channels"]) == 4, "advantage encoder input channels must be 4")


def _check_reward_ablation_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/final_method/run_a_new_reward_ablation_batch.py",
            "--reward-ablation-ids",
            "R1,R2,R3,R4,R5",
            "--run-stage",
            "smoke",
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    payload = _load_json_from_output(result.stdout)
    expected_overrides = {
        "Anew_R1": {"reward_step_penalty": 0.0},
        "Anew_R2": {"reward_revisit_penalty": 0.0},
        "Anew_R3": {"reward_turn_penalty_scale": 0.0},
        "Anew_R4": {"reward_timeout_penalty": 0.0},
        "Anew_R5": {
            "reward_step_penalty": 0.0,
            "reward_revisit_penalty": 0.0,
            "reward_turn_penalty_scale": 0.0,
            "reward_timeout_penalty": 0.0,
        },
    }
    _assert(payload["baseline_method"] == "A_new_final_4ch_no_frontier_raster", "baseline method mismatch")
    methods = payload["methods"]
    _assert(isinstance(methods, list) and len(methods) == 5, "expected five Anew_R dry-run methods")
    for method in methods:
        method_id = str(method["method_id"])
        _assert(method["advantage_canvas_schema"] == ADVANTAGE_CANVAS_SCHEMA_FINAL_4CH_NO_FRONTIER_RASTER, f"{method_id} schema mismatch")
        _assert(method["advantage_canvas_channels"] == list(FINAL_4CH_ADVANTAGE_CANVAS_CHANNELS), f"{method_id} channel list mismatch")
        _assert(int(method["advantage_canvas_channel_count"]) == 4, f"{method_id} channel count must be 4")
        _assert(method["frontier_raster_used"] is False, f"{method_id} frontier_raster_used must be false")
        _assert(method["value_tree_enabled"] is True, f"{method_id} value tree must stay enabled")
        _assert(int(method["advantage_encoder.canvas_in_channels"]) == 4, f"{method_id} encoder channels must be 4")
        _assert(method["reward_override"] == expected_overrides[method_id], f"{method_id} reward override mismatch")


def main() -> int:
    _check_final_canvas()
    _check_network_and_value_tree()
    _check_a_new_final_dry_run()
    _check_reward_ablation_dry_run()
    print("A_new final 4-channel checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
