from __future__ import annotations

"""
导出方法示意图所需的静态图片。

运行方式：
    python tools/export_architecture_pictures.py

输出目录：
    run_picture/
"""

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import font_manager
from matplotlib.patches import Circle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, GridTopology


def _configure_matplotlib_chinese_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available_names = [font.name for font in font_manager.fontManager.ttflist]
    chosen: list[str] = []
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for available in available_names:
            available_lower = available.lower()
            if candidate_lower == available_lower or candidate_lower in available_lower:
                if available not in chosen:
                    chosen.append(available)
                break

    if chosen:
        existing = list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = chosen + [name for name in existing if name not in chosen]
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_chinese_fonts()

BELIEF_CMAP = ListedColormap(
    [
        "#5f6770",  # unknown
        "#f5f6f7",  # free
        "#1c232b",  # obstacle
    ]
)
BELIEF_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], BELIEF_CMAP.N)

AGENT_COLOR = "#f2542d"
AGENT_MARKER_SIZE = 74
AGENT_EDGE_COLOR = "white"
AGENT_EDGE_WIDTH = 1.1
SCAN_EDGE_COLOR = "#0f4c5c"
TRAJECTORY_COLOR = "#2d6a8c"
TRAJECTORY_LINEWIDTH = 1.8

KEY_TO_ACTION = {
    "w": 0,
    "e": 1,
    "d": 2,
    "c": 3,
    "x": 4,
    "z": 5,
    "a": 6,
    "q": 7,
}
ACTION_TO_KEY = {action_idx: key for key, action_idx in KEY_TO_ACTION.items()}

FIXED_ACTION_PREFERENCES = (
    "q",
    "w",
    "w",
    "e",
    "w",
    "d",
    "d",
    "e",
)


@dataclass(frozen=True)
class ExportConfig:
    rows: int = 40
    cols: int = 60
    obstacle_ratio: float = 0.20
    obs_size: int = 6
    scan_radius: int = 10
    seed: int = 0
    step_mid: int = 4
    step_late: int = 8
    dpi: int = 240
    output_dir: Path = REPO_ROOT / "run_picture"


@dataclass(frozen=True)
class Snapshot:
    step: int
    agent_world: tuple[int, int]
    agent_array: tuple[int, int]
    trajectory_array: np.ndarray
    local_snap: np.ndarray
    belief_map: np.ndarray


def _set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))


def _clear_old_png_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for png_path in output_dir.glob("*.png"):
        png_path.unlink()


def _capture_snapshot(
    *,
    step: int,
    agent_state: tuple[int, int],
    trajectory_world: list[tuple[int, int]],
    local_snap: np.ndarray,
    cum_map: CumulativeBeliefMap,
) -> Snapshot:
    return Snapshot(
        step=int(step),
        agent_world=(int(agent_state[0]), int(agent_state[1])),
        agent_array=tuple(int(v) for v in cum_map.world_to_array(agent_state)),
        trajectory_array=np.asarray(
            [tuple(int(v) for v in cum_map.world_to_array(world_rc)) for world_rc in trajectory_world],
            dtype=np.int32,
        ),
        local_snap=np.asarray(local_snap, dtype=np.int8).copy(),
        belief_map=np.asarray(cum_map.map, dtype=np.int8).copy(),
    )


def _select_fallback_action(
    valid_actions: tuple[int, ...],
    *,
    agent_state: tuple[int, int],
    visit_counts: dict[tuple[int, int], int],
) -> int:
    return min(
        valid_actions,
        key=lambda action_idx: (
            visit_counts.get(
                (
                    int(agent_state[0] + ACTIONS_8[action_idx][0]),
                    int(agent_state[1] + ACTIONS_8[action_idx][1]),
                ),
                0,
            ),
            int(action_idx),
        ),
    )


def _run_deterministic_rollout(
    config: ExportConfig,
) -> tuple[RadarSensor, dict[int, Snapshot], tuple[str, ...], tuple[str, ...]]:
    _set_global_seed(config.seed)

    generator = RandomMapGenerator(
        rows=int(config.rows),
        cols=int(config.cols),
        obs_size=int(config.obs_size),
        obstacle_ratio=float(config.obstacle_ratio),
    )
    true_grid, start_state = generator.generate_map()
    free_mask = GridTopology.free_mask(true_grid)

    sensor = RadarSensor(scan_radius=int(config.scan_radius))
    obs_model = LocalObservationModel(true_grid, start_state, sensor=sensor)
    local_snap = np.asarray(obs_model.local_snap, dtype=np.int8).copy()
    cum_map = CumulativeBeliefMap(true_grid, start_state, local_snap)

    checkpoints = {0, int(config.step_mid), int(config.step_late)}
    agent_state = (int(start_state[0]), int(start_state[1]))
    trajectory_world = [agent_state]
    snapshots: dict[int, Snapshot] = {
        0: _capture_snapshot(
            step=0,
            agent_state=agent_state,
            trajectory_world=trajectory_world,
            local_snap=local_snap,
            cum_map=cum_map,
        )
    }
    visit_counts: dict[tuple[int, int], int] = {agent_state: 1}
    executed_keys: list[str] = []

    for step_idx in range(1, int(config.step_late) + 1):
        planned_key = FIXED_ACTION_PREFERENCES[(step_idx - 1) % len(FIXED_ACTION_PREFERENCES)]
        desired_action = int(KEY_TO_ACTION[planned_key])
        valid_actions = GridTopology.valid_action_indices_fast(free_mask, agent_state)
        if not valid_actions:
            raise RuntimeError(f"agent has no legal moves at step {step_idx}")

        if desired_action in valid_actions:
            chosen_action = desired_action
        else:
            chosen_action = _select_fallback_action(
                valid_actions,
                agent_state=agent_state,
                visit_counts=visit_counts,
            )

        dr, dc = ACTIONS_8[chosen_action]
        agent_state = (int(agent_state[0] + dr), int(agent_state[1] + dc))
        visit_counts[agent_state] = int(visit_counts.get(agent_state, 0) + 1)
        trajectory_world.append(agent_state)

        local_snap = np.asarray(obs_model.observe_fast(agent_state), dtype=np.int8).copy()
        cum_map.update(agent_state, local_snap)
        executed_keys.append(ACTION_TO_KEY[chosen_action])

        if step_idx in checkpoints:
            snapshots[step_idx] = _capture_snapshot(
                step=step_idx,
                agent_state=agent_state,
                trajectory_world=trajectory_world,
                local_snap=local_snap,
                cum_map=cum_map,
            )

    missing_steps = [step for step in sorted(checkpoints) if step not in snapshots]
    if missing_steps:
        raise RuntimeError(f"missing rollout checkpoints: {missing_steps}")

    if len(executed_keys) != int(config.step_late):
        raise RuntimeError("rollout did not produce the requested number of effective moves")

    return sensor, snapshots, tuple(FIXED_ACTION_PREFERENCES), tuple(executed_keys)


def _format_clean_axis(ax, shape: tuple[int, int]) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, float(shape[1]) - 0.5)
    ax.set_ylim(float(shape[0]) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_local_axis(ax, snapshot: Snapshot, sensor: RadarSensor) -> None:
    ax.imshow(snapshot.local_snap, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")

    center_r, center_c = int(sensor.center_state[0]), int(sensor.center_state[1])
    ax.add_patch(
        Circle(
            (float(center_c), float(center_r)),
            radius=float(sensor.scan_r) + 0.15,
            fill=False,
            edgecolor=SCAN_EDGE_COLOR,
            linewidth=1.4,
            linestyle="--",
            alpha=0.90,
            zorder=4,
        )
    )
    ax.scatter(
        [float(center_c)],
        [float(center_r)],
        marker="o",
        s=AGENT_MARKER_SIZE,
        c=AGENT_COLOR,
        edgecolors=AGENT_EDGE_COLOR,
        linewidths=AGENT_EDGE_WIDTH,
        zorder=5,
    )
    _format_clean_axis(ax, snapshot.local_snap.shape)


def _render_belief_axis(ax, snapshot: Snapshot) -> None:
    ax.imshow(snapshot.belief_map, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")

    if snapshot.trajectory_array.shape[0] > 1:
        ax.plot(
            snapshot.trajectory_array[:, 1].astype(np.float32),
            snapshot.trajectory_array[:, 0].astype(np.float32),
            color=TRAJECTORY_COLOR,
            linewidth=TRAJECTORY_LINEWIDTH,
            alpha=0.96,
            solid_capstyle="round",
            zorder=4,
        )

    ax.scatter(
        [float(snapshot.agent_array[1])],
        [float(snapshot.agent_array[0])],
        marker="o",
        s=AGENT_MARKER_SIZE,
        c=AGENT_COLOR,
        edgecolors=AGENT_EDGE_COLOR,
        linewidths=AGENT_EDGE_WIDTH,
        zorder=5,
    )
    _format_clean_axis(ax, snapshot.belief_map.shape)


def _grid_figure_size(shape: tuple[int, int], *, height: float = 4.8, min_width: float = 4.0) -> tuple[float, float]:
    rows, cols = int(shape[0]), int(shape[1])
    width = max(float(min_width), float(height) * (float(cols) / max(float(rows), 1.0)))
    return width, float(height)


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def _export_local_radar_observation(path: Path, snapshot: Snapshot, sensor: RadarSensor, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=_grid_figure_size(snapshot.local_snap.shape, height=4.4, min_width=4.4))
    _render_local_axis(ax, snapshot, sensor)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save_figure(fig, path, dpi=dpi)


def _export_belief_map(path: Path, snapshot: Snapshot, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=_grid_figure_size(snapshot.belief_map.shape, height=4.9, min_width=4.2))
    _render_belief_axis(ax, snapshot)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    _save_figure(fig, path, dpi=dpi)


def _export_belief_growth_montage(
    path: Path,
    *,
    step0: Snapshot,
    step_mid: Snapshot,
    step_late: Snapshot,
    dpi: int,
) -> None:
    panels = ((step0, "初始步"), (step_mid, "第4步"), (step_late, "第8步"))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.4, 4.8),
        gridspec_kw={"width_ratios": [panel.belief_map.shape[1] for panel, _ in panels]},
    )

    for ax, (snapshot, label) in zip(np.ravel(axes), panels):
        _render_belief_axis(ax, snapshot)
        ax.text(
            0.5,
            -0.08,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#243b53",
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.16, wspace=0.18)
    _save_figure(fig, path, dpi=dpi)


def _export_local_to_belief_pair(
    path: Path,
    *,
    local_snapshot: Snapshot,
    belief_snapshot: Snapshot,
    sensor: RadarSensor,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.8),
        gridspec_kw={
            "width_ratios": [
                int(local_snapshot.local_snap.shape[1]),
                int(belief_snapshot.belief_map.shape[1]),
            ]
        },
    )

    _render_local_axis(axes[0], local_snapshot, sensor)
    _render_belief_axis(axes[1], belief_snapshot)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.34)
    _save_figure(fig, path, dpi=dpi)


def main() -> None:
    config = ExportConfig()
    _clear_old_png_outputs(config.output_dir)

    sensor, snapshots, planned_keys, executed_keys = _run_deterministic_rollout(config)

    step0_snapshot = snapshots[0]
    step_mid_snapshot = snapshots[int(config.step_mid)]
    step_late_snapshot = snapshots[int(config.step_late)]

    outputs = [
        config.output_dir / "局部雷达观测.png",
        config.output_dir / "累计认知地图_初始步.png",
        config.output_dir / "累计认知地图_第4步.png",
        config.output_dir / "累计认知地图_第8步.png",
        config.output_dir / "累计认知地图动态增长.png",
        config.output_dir / "局部观测到累计认知地图.png",
    ]

    _export_local_radar_observation(outputs[0], step_late_snapshot, sensor, dpi=config.dpi)
    _export_belief_map(outputs[1], step0_snapshot, dpi=config.dpi)
    _export_belief_map(outputs[2], step_mid_snapshot, dpi=config.dpi)
    _export_belief_map(outputs[3], step_late_snapshot, dpi=config.dpi)
    _export_belief_growth_montage(
        outputs[4],
        step0=step0_snapshot,
        step_mid=step_mid_snapshot,
        step_late=step_late_snapshot,
        dpi=config.dpi,
    )
    _export_local_to_belief_pair(
        outputs[5],
        local_snapshot=step_late_snapshot,
        belief_snapshot=step_late_snapshot,
        sensor=sensor,
        dpi=config.dpi,
    )

    print(f"seed={config.seed}")
    print(f"action_preferences={' '.join(planned_keys)}")
    print(f"executed_actions={' '.join(executed_keys)}")
    print(
        "fallback_rule=当预设动作不合法时，从当前合法动作中选择“下一位置访问次数最少、若并列则动作索引最小”的稳定备选"
    )
    print(f"effective_moves={len(executed_keys)}")
    print(f"checkpoints=0,{config.step_mid},{config.step_late}")
    for output in outputs:
        print(output.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
