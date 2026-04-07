from __future__ import annotations

"""
实时交互式语义探索演示 demo。

运行方式：
    python demos/interactive_semantic_demo.py

按键说明：
    q / w / e : 左上 / 上 / 右上
    a / d     : 左 / 右
    z / x / c : 左下 / 下 / 右下
    r         : 重置当前地图到初始状态
    n         : 生成新随机地图
    p         : 切换语义叠加显示
    b         : 切换可达未知块显示
    f         : 切换前沿簇显示
    t         : 切换轨迹显示
    i         : 切换前沿簇编号显示
    s         : 保存截图到 outputs/demo_frames/
    h         : 打印帮助信息
    esc       : 退出
"""

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Rectangle

from env.agent_version import LocalObservationModel
from env.block_random_g import RandomMapGenerator
from env.core_cummap import CumulativeBeliefMap
from env.core_radar import RadarSensor
from env.grid_topology import ACTIONS_8, INVISIBLE, GridTopology
from env.shared_semantic_layer import SharedSemanticLayer, build_semantic_visualization_payload

MOVE_KEY_TO_ACTION = {
    "w": 0,  # N
    "e": 1,  # NE
    "d": 2,  # E
    "c": 3,  # SE
    "x": 4,  # S
    "z": 5,  # SW
    "a": 6,  # W
    "q": 7,  # NW
}

MOVE_KEY_ORDER = ("q", "w", "e", "a", "d", "z", "x", "c")
ACTION_TO_KEY = {action_idx: key for key, action_idx in MOVE_KEY_TO_ACTION.items()}
ACTION_TO_NAME = {
    0: "上(N)",
    1: "右上(NE)",
    2: "右(E)",
    3: "右下(SE)",
    4: "下(S)",
    5: "左下(SW)",
    6: "左(W)",
    7: "左上(NW)",
}

BELIEF_CMAP = ListedColormap(
    [
        "#5f6770",  # unknown
        "#f5f6f7",  # free
        "#1c232b",  # obstacle
    ]
)
BELIEF_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], BELIEF_CMAP.N)

TRUE_MAP_CMAP = ListedColormap(
    [
        "#f7f6f2",  # free
        "#15181d",  # obstacle
    ]
)
TRUE_MAP_NORM = BoundaryNorm([-0.5, 0.5, 1.5], TRUE_MAP_CMAP.N)

BLOCK_CMAP = plt.get_cmap("tab20")
ENTRY_CMAP = plt.get_cmap("Set2")


@dataclass(frozen=True)
class DemoConfig:
    rows: int = 40
    cols: int = 60
    obstacle_ratio: float = 0.20
    obs_size: int = 6
    scan_radius: int = 10
    seed: Optional[int] = None
    show_true_map: bool = True
    show_semantics: bool = True
    show_local_observation: bool = True
    show_help_on_start: bool = True
    screenshot_dir: Path = REPO_ROOT / "outputs" / "demo_frames"


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


def _mix_color(rgb_a: np.ndarray, rgb_b: np.ndarray, weight_b: float) -> np.ndarray:
    weight = float(np.clip(weight_b, 0.0, 1.0))
    return ((1.0 - weight) * rgb_a) + (weight * rgb_b)


def _rgb_from_cmap(cmap, index: int) -> np.ndarray:
    return np.asarray(cmap(int(index) % int(cmap.N))[:3], dtype=np.float32)


def _safe_metric_text(name: str, value: float) -> str:
    if not np.isfinite(value):
        return f"{name}: n/a"
    if abs(value - round(value)) <= 1e-6 and abs(value) < 1e6:
        return f"{name}: {int(round(value))}"
    return f"{name}: {value:.3f}"


def print_controls() -> None:
    print("")
    print("实时交互式语义探索演示 Demo 按键说明")
    print("------------------------------------")
    print("移动：q w e / a d / z x c")
    print("  q=左上(NW)  w=上(N)  e=右上(NE)")
    print("  a=左(W)               d=右(E)")
    print("  z=左下(SW)  x=下(S)  c=右下(SE)")
    print("r = 重置当前地图")
    print("n = 生成新地图")
    print("p = 切换语义叠加显示")
    print("b = 切换未知块显示")
    print("f = 切换前沿簇显示")
    print("t = 切换轨迹显示")
    print("i = 切换前沿簇编号显示")
    print("s = 保存截图")
    print("h = 打印帮助")
    print("esc 或关闭窗口 = 退出")
    print("")


class InteractiveSemanticDemo:
    def __init__(self, config: DemoConfig):
        self.config = config
        if self.config.seed is not None:
            np.random.seed(int(self.config.seed))
            random.seed(int(self.config.seed))

        self.generator = RandomMapGenerator(
            rows=int(self.config.rows),
            cols=int(self.config.cols),
            obs_size=int(self.config.obs_size),
            obstacle_ratio=float(self.config.obstacle_ratio),
        )
        self.sensor = RadarSensor(scan_radius=int(self.config.scan_radius))
        self.shared_semantic_layer = SharedSemanticLayer()

        self.show_semantics = bool(self.config.show_semantics)
        self.show_blocks = True
        self.show_frontiers = True
        self.show_trajectory = True
        self.show_entry_labels = True

        self.status_message = "就绪"
        self.status_is_error = False
        self.map_generation_index = 0

        self.true_grid: np.ndarray | None = None
        self.free_mask: np.ndarray | None = None
        self.start_state: tuple[int, int] | None = None
        self.agent_state: tuple[int, int] | None = None
        self.obs_model: LocalObservationModel | None = None
        self.local_snap: np.ndarray | None = None
        self.cum_map: CumulativeBeliefMap | None = None
        self.semantic_snapshot = None
        self.semantic_payload: dict[str, object] | None = None
        self.valid_action_indices: tuple[int, ...] = ()
        self.trajectory_world: list[tuple[int, int]] = []
        self.trajectory_array: list[tuple[int, int]] = []
        self.visible_world_rows = np.zeros((0,), dtype=np.int32)
        self.visible_world_cols = np.zeros((0,), dtype=np.int32)
        self.frontier_u8 = np.zeros((1, 1), dtype=np.uint8)
        self.metrics: dict[str, float] = {}
        self.entry_count = 0
        self.agent_array = (0, 0)

        self._base_grid: np.ndarray | None = None
        self._base_start_state: tuple[int, int] | None = None

        self.figure = None
        self.axes: dict[str, object] = {}

        self.new_map(initial=True)

    def _set_status(self, message: str, *, is_error: bool = False, print_to_terminal: bool = False) -> None:
        self.status_message = str(message)
        self.status_is_error = bool(is_error)
        if print_to_terminal:
            print(self.status_message)

    def _demo_step(self) -> int:
        return max(0, int(len(self.trajectory_world) - 1))

    def _visible_world_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.agent_state is not None
        assert self.local_snap is not None
        assert self.true_grid is not None

        gr, gc = GridTopology.local_to_global_grid(
            self.agent_state,
            tuple(self.local_snap.shape),
            self.sensor.center_state,
        )
        inside = (
            (gr >= 0)
            & (gc >= 0)
            & (gr < int(self.true_grid.shape[0]))
            & (gc < int(self.true_grid.shape[1]))
        )
        visible = inside & (self.local_snap != INVISIBLE)
        return (
            np.asarray(gr[visible], dtype=np.int32),
            np.asarray(gc[visible], dtype=np.int32),
        )

    def _refresh_runtime_views(self) -> None:
        assert self.true_grid is not None
        assert self.free_mask is not None
        assert self.agent_state is not None
        assert self.cum_map is not None

        self.valid_action_indices = GridTopology.valid_action_indices_fast(self.free_mask, self.agent_state)
        self.frontier_u8 = np.asarray(self.cum_map.get_frontier_u8(refresh=False), dtype=np.uint8).copy()
        self.semantic_snapshot = self.shared_semantic_layer.analyze(self.cum_map, self.agent_state)
        self.semantic_payload = build_semantic_visualization_payload(self.semantic_snapshot)
        self.metrics = dict(self.semantic_snapshot.metrics())
        self.entry_count = sum(len(block["frontier_clusters"]) for block in self.semantic_payload["blocks"])
        self.agent_array = tuple(int(v) for v in self.cum_map.world_to_array(self.agent_state))
        self.trajectory_array = [
            tuple(int(v) for v in self.cum_map.world_to_array(world_rc))
            for world_rc in self.trajectory_world
        ]
        self.visible_world_rows, self.visible_world_cols = self._visible_world_coordinates()

    def _load_episode(self, grid: np.ndarray, start_state: tuple[int, int], *, status_message: str) -> None:
        self.true_grid = np.asarray(grid, dtype=np.int8).copy()
        self.start_state = (int(start_state[0]), int(start_state[1]))
        self.agent_state = self.start_state
        self.free_mask = GridTopology.free_mask(self.true_grid)

        self.obs_model = LocalObservationModel(self.true_grid, self.agent_state, sensor=self.sensor)
        self.local_snap = np.asarray(self.obs_model.local_snap, dtype=np.int8).copy()
        self.cum_map = CumulativeBeliefMap(self.true_grid, self.agent_state, self.local_snap)

        self.trajectory_world = [self.agent_state]
        self._refresh_runtime_views()
        self._set_status(status_message)

    def new_map(self, *, initial: bool = False) -> None:
        grid, start_state = self.generator.generate_map()
        self._base_grid = np.asarray(grid, dtype=np.int8).copy()
        self._base_start_state = (int(start_state[0]), int(start_state[1]))
        self.map_generation_index += 1
        prefix = "initialized" if initial else "generated new map"
        if initial:
            prefix = "初始化完成"
        else:
            prefix = "已生成新地图"
        self._load_episode(self._base_grid, self._base_start_state, status_message=prefix)

    def reset_current_map(self) -> None:
        if self._base_grid is None or self._base_start_state is None:
            self.new_map()
            return
        self._load_episode(self._base_grid, self._base_start_state, status_message="已重置当前地图")

    def save_screenshot(self) -> Path:
        if self.figure is None:
            raise RuntimeError("图形窗口尚未创建")
        assert self.cum_map is not None
        self.config.screenshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.screenshot_dir / (
            f"interactive_semantic_demo_map{self.map_generation_index:03d}_step{self._demo_step():04d}_{timestamp}.png"
        )
        self.figure.savefig(path, dpi=160, bbox_inches="tight")
        self._set_status(f"已保存截图：{path}")
        return path

    def step_with_action_index(self, action_idx: int, *, key: Optional[str] = None) -> bool:
        assert self.agent_state is not None
        assert self.obs_model is not None
        assert self.cum_map is not None

        action_idx = int(action_idx)
        valid_set = set(self.valid_action_indices)
        key_name = key if key is not None else ACTION_TO_KEY.get(action_idx, "?")
        if action_idx not in valid_set:
            self._set_status(
                f"非法动作：{key_name} -> {ACTION_TO_NAME.get(action_idx, '?')}",
                is_error=True,
                print_to_terminal=True,
            )
            return False

        dr, dc = ACTIONS_8[action_idx]
        self.agent_state = (int(self.agent_state[0] + dr), int(self.agent_state[1] + dc))
        self.trajectory_world.append(self.agent_state)

        snap = self.obs_model.observe_fast(self.agent_state)
        self.local_snap = np.asarray(snap, dtype=np.int8).copy()
        updated, delta_empty, delta_obstacle = self.cum_map.update(self.agent_state, self.local_snap)
        if int(updated) != int(delta_empty + delta_obstacle):
            raise RuntimeError("belief-map update returned inconsistent information-gain counts")

        self._refresh_runtime_views()
        self._set_status(
            (
                f"步数 {self._demo_step()}：{key_name} -> {ACTION_TO_NAME[action_idx]} | "
                f"新揭示 {int(updated)} 个栅格（空闲 {int(delta_empty)}，障碍 {int(delta_obstacle)}）"
            )
        )
        return True

    def _setup_figure(self) -> None:
        if self.figure is not None:
            return
        self.figure, axs = plt.subplot_mosaic(
            [["true", "belief"], ["semantic", "local"]],
            figsize=(16, 11),
            constrained_layout=True,
        )
        self.axes = dict(axs)
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("实时交互式语义探索演示 Demo")
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _event) -> None:
        self._set_status("窗口已关闭")

    def _on_key_press(self, event) -> None:
        if event.key is None:
            return
        key = str(event.key).lower()

        if key == "escape":
            plt.close(self.figure)
            return
        if key in MOVE_KEY_TO_ACTION:
            self.step_with_action_index(MOVE_KEY_TO_ACTION[key], key=key)
        elif key == "r":
            self.reset_current_map()
        elif key == "n":
            self.new_map()
        elif key == "p":
            self.show_semantics = not self.show_semantics
            self._set_status(f"语义叠加显示已{'开启' if self.show_semantics else '关闭'}")
        elif key == "b":
            self.show_blocks = not self.show_blocks
            self._set_status(f"未知块显示已{'开启' if self.show_blocks else '关闭'}")
        elif key == "f":
            self.show_frontiers = not self.show_frontiers
            self._set_status(f"前沿簇显示已{'开启' if self.show_frontiers else '关闭'}")
        elif key == "t":
            self.show_trajectory = not self.show_trajectory
            self._set_status(f"轨迹显示已{'开启' if self.show_trajectory else '关闭'}")
        elif key == "i":
            self.show_entry_labels = not self.show_entry_labels
            self._set_status(f"前沿簇编号显示已{'开启' if self.show_entry_labels else '关闭'}")
        elif key == "h":
            print_controls()
            self._set_status("已在终端打印帮助信息")
        elif key == "s":
            path = self.save_screenshot()
            print(f"截图已保存到：{path}")
        else:
            return

        self.render()

    @staticmethod
    def _format_axis(ax, shape: tuple[int, int], *, title: str) -> None:
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, float(shape[1]) - 0.5)
        ax.set_ylim(float(shape[0]) - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    def _trajectory_xy_world(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.trajectory_world) <= 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        pts = np.asarray(self.trajectory_world, dtype=np.int32)
        return pts[:, 1].astype(np.float32), pts[:, 0].astype(np.float32)

    def _trajectory_xy_array(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.trajectory_array) <= 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        pts = np.asarray(self.trajectory_array, dtype=np.int32)
        return pts[:, 1].astype(np.float32), pts[:, 0].astype(np.float32)

    @staticmethod
    def _semantic_block_bbox(rows: np.ndarray, cols: np.ndarray) -> tuple[float, float, float, float]:
        r0 = float(np.min(rows)) - 0.5
        r1 = float(np.max(rows)) + 0.5
        c0 = float(np.min(cols)) - 0.5
        c1 = float(np.max(cols)) + 0.5
        return r0, r1, c0, c1

    def _draw_true_map_panel(self, ax) -> None:
        assert self.true_grid is not None
        assert self.agent_state is not None
        assert self.start_state is not None

        ax.clear()
        if not self.config.show_true_map:
            ax.text(0.5, 0.5, "真值地图面板已隐藏", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        ax.imshow(self.true_grid, cmap=TRUE_MAP_CMAP, norm=TRUE_MAP_NORM, origin="upper", interpolation="nearest")

        footprint_rgba = np.zeros((*self.true_grid.shape, 4), dtype=np.float32)
        footprint_rgba[self.visible_world_rows, self.visible_world_cols] = np.array([0.12, 0.72, 0.84, 0.32], dtype=np.float32)
        ax.imshow(footprint_rgba, origin="upper", interpolation="nearest")

        if self.show_trajectory and len(self.trajectory_world) > 1:
            traj_x, traj_y = self._trajectory_xy_world()
            ax.plot(traj_x, traj_y, color="#d84f35", linewidth=1.6, alpha=0.90, zorder=4)

        ax.scatter(
            [int(self.start_state[1])],
            [int(self.start_state[0])],
            marker="x",
            s=70,
            c="#226f54",
            linewidths=2.0,
            zorder=5,
        )
        ax.scatter(
            [int(self.agent_state[1])],
            [int(self.agent_state[0])],
            marker="o",
            s=90,
            c="#f2542d",
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
        )

        visible_text = f"当前可见：{int(len(self.visible_world_rows))}/{int(self.sensor.theoretical_visible_cell_count)}"
        pose_text = f"智能体（世界坐标）：({int(self.agent_state[0])}, {int(self.agent_state[1])})"
        ax.text(
            0.01,
            0.99,
            f"{pose_text}\n{visible_text}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffffff2", edgecolor="#999999"),
        )
        self._format_axis(ax, self.true_grid.shape, title="真值地图（True Map）")

    def _draw_belief_panel(self, ax) -> None:
        assert self.cum_map is not None

        ax.clear()
        belief_map = np.asarray(self.cum_map.map, dtype=np.int8)
        ax.imshow(belief_map, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")

        frontier_mask = self.frontier_u8 > 0
        frontier_rgba = np.zeros((*belief_map.shape, 4), dtype=np.float32)
        frontier_rgba[frontier_mask] = np.array([0.96, 0.73, 0.16, 0.62], dtype=np.float32)
        ax.imshow(frontier_rgba, origin="upper", interpolation="nearest")

        if self.show_trajectory and len(self.trajectory_array) > 1:
            traj_x, traj_y = self._trajectory_xy_array()
            ax.plot(traj_x, traj_y, color="#cb4b16", linewidth=1.6, alpha=0.90, zorder=4)

        valid_set = set(self.valid_action_indices)
        for key in MOVE_KEY_ORDER:
            action_idx = MOVE_KEY_TO_ACTION[key]
            dr, dc = ACTIONS_8[action_idx]
            target_world = (int(self.agent_state[0] + dr), int(self.agent_state[1] + dc))
            target_arr = self.cum_map.world_to_array(target_world)
            tr, tc = int(target_arr[0]), int(target_arr[1])
            if not (0 <= tr < belief_map.shape[0] and 0 <= tc < belief_map.shape[1]):
                continue
            is_valid = action_idx in valid_set
            edge_color = "#2a9d8f" if is_valid else "#b23a48"
            face_color = "#ffffff" if is_valid else "none"
            ax.scatter(
                [tc],
                [tr],
                marker="s",
                s=95,
                facecolors=face_color,
                edgecolors=edge_color,
                linewidths=1.4,
                alpha=0.90,
                zorder=5,
            )
            ax.text(
                tc,
                tr,
                key.upper(),
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                color=edge_color,
                zorder=6,
            )

        ax.scatter(
            [int(self.agent_array[1])],
            [int(self.agent_array[0])],
            marker="o",
            s=90,
            c="#f2542d",
            edgecolors="white",
            linewidths=1.2,
            zorder=7,
        )

        valid_keys = [ACTION_TO_KEY[idx] for idx in self.valid_action_indices]
        belief_text = "\n".join(
            [
                f"智能体（数组坐标）：({int(self.agent_array[0])}, {int(self.agent_array[1])})",
                f"覆盖率：{float(self.cum_map.coverage_rate):.1%}",
                f"合法动作[{len(self.valid_action_indices)}]：{' '.join(valid_keys) if valid_keys else '无'}",
                f"前沿栅格数：{int(np.count_nonzero(frontier_mask))}",
            ]
        )
        ax.text(
            0.01,
            0.99,
            belief_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffffff2", edgecolor="#999999"),
        )
        self._format_axis(ax, belief_map.shape, title="累计 Belief 地图（Cumulative Belief Map）")

    def _draw_semantic_panel(self, ax) -> None:
        assert self.cum_map is not None
        assert self.semantic_payload is not None

        ax.clear()
        belief_map = np.asarray(self.cum_map.map, dtype=np.int8)
        ax.imshow(belief_map, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")

        analysis_box = self.semantic_payload["analysis_box"]
        ax.add_patch(
            Rectangle(
                (float(analysis_box["c0"]) - 0.5, float(analysis_box["r0"]) - 0.5),
                float(analysis_box["c1"] - analysis_box["c0"]),
                float(analysis_box["r1"] - analysis_box["r0"]),
                fill=False,
                edgecolor="#0f4c5c",
                linewidth=2.0,
                linestyle="--",
                zorder=8,
            )
        )

        if self.show_semantics:
            blocks = self.semantic_payload["blocks"]
            block_rgba = np.zeros((*belief_map.shape, 4), dtype=np.float32)
            support_rgba = np.zeros((*belief_map.shape, 4), dtype=np.float32)
            frontier_rgba = np.zeros((*belief_map.shape, 4), dtype=np.float32)
            block_labels: list[tuple[float, float, str]] = []
            frontier_labels: list[tuple[float, float, str]] = []
            support_boxes: list[tuple[float, float, float, float, np.ndarray]] = []

            for block_slot, block in enumerate(blocks):
                block_rows = np.asarray(block["rows"], dtype=np.int32)
                block_cols = np.asarray(block["cols"], dtype=np.int32)
                if block_rows.size <= 0 or block_cols.size <= 0:
                    continue

                block_rgb = _rgb_from_cmap(BLOCK_CMAP, int(block_slot))
                block_alpha = 0.36
                if self.show_blocks:
                    block_rgba[block_rows, block_cols, :3] = block_rgb
                    block_rgba[block_rows, block_cols, 3] = np.maximum(
                        block_rgba[block_rows, block_cols, 3],
                        block_alpha,
                    )

                if self.show_blocks:
                    label_r = float(np.mean(block_rows))
                    label_c = float(np.mean(block_cols))
                    block_labels.append((label_c, label_r, f"B{int(block['block_index'])}"))

                if self.show_frontiers:
                    for frontier_slot, frontier_cluster in enumerate(block["frontier_clusters"]):
                        frontier_rows = np.asarray(frontier_cluster["frontier_rows"], dtype=np.int32)
                        frontier_cols = np.asarray(frontier_cluster["frontier_cols"], dtype=np.int32)
                        support = frontier_cluster["support"]
                        support_rows = np.asarray(support["free_rows"], dtype=np.int32)
                        support_cols = np.asarray(support["free_cols"], dtype=np.int32)
                        support_box = support["local_box"]
                        entry_rgb = _mix_color(
                            block_rgb,
                            _rgb_from_cmap(ENTRY_CMAP, int(frontier_slot)),
                            0.35,
                        )
                        support_rgba[support_rows, support_cols, :3] = entry_rgb
                        support_rgba[support_rows, support_cols, 3] = np.maximum(
                            support_rgba[support_rows, support_cols, 3],
                            0.46,
                        )
                        frontier_rgba[frontier_rows, frontier_cols, :3] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                        frontier_rgba[frontier_rows, frontier_cols, 3] = 1.0
                        support_boxes.append(
                            (
                                float(support_box["c0"]) - 0.5,
                                float(support_box["r0"]) - 0.5,
                                float(support_box["c1"] - support_box["c0"]),
                                float(support_box["r1"] - support_box["r0"]),
                                entry_rgb,
                            )
                        )

                        anchor_r, anchor_c = frontier_cluster["frontier_anchor_rc"]
                        label_rows = frontier_rows
                        label_cols = frontier_cols
                        if label_rows.size > 0 and label_cols.size > 0:
                            frontier_labels.append(
                                (
                                    float(anchor_c),
                                    float(anchor_r),
                                    f"F{int(frontier_cluster['frontier_index'])}",
                                )
                            )

            ax.imshow(block_rgba, origin="upper", interpolation="nearest")
            if self.show_frontiers:
                ax.imshow(support_rgba, origin="upper", interpolation="nearest")
                ax.imshow(frontier_rgba, origin="upper", interpolation="nearest")
                for box_c0, box_r0, box_w, box_h, box_rgb in support_boxes:
                    ax.add_patch(
                        Rectangle(
                            (box_c0, box_r0),
                            box_w,
                            box_h,
                            fill=False,
                            edgecolor=box_rgb,
                            linewidth=1.1,
                            linestyle=":",
                            alpha=0.75,
                            zorder=8,
                        )
                    )

            for label_c, label_r, text in block_labels:
                ax.text(
                    label_c,
                    label_r,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="#00000099",
                        edgecolor="#ffffff80",
                    ),
                    zorder=10,
                )

            if self.show_frontiers and self.show_entry_labels:
                for label_c, label_r, text in frontier_labels:
                    ax.text(
                        label_c,
                        label_r,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#102a43",
                        bbox=dict(
                            boxstyle="round,pad=0.12",
                            facecolor="#ffffffdd",
                            edgecolor="#4f6d7a",
                        ),
                        zorder=10,
                    )
        else:
            ax.text(
                0.5,
                0.5,
                "语义叠加已隐藏（按 'p' 切换）",
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffffff2", edgecolor="#999999"),
                zorder=10,
            )

        if self.show_trajectory and len(self.trajectory_array) > 1:
            traj_x, traj_y = self._trajectory_xy_array()
            ax.plot(traj_x, traj_y, color="#bc3908", linewidth=1.5, alpha=0.90, zorder=6)

        ax.scatter(
            [int(self.agent_array[1])],
            [int(self.agent_array[0])],
            marker="o",
            s=92,
            c="#f2542d",
            edgecolors="white",
            linewidths=1.2,
            zorder=11,
        )

        semantic_text = "\n".join(
            [
                _safe_metric_text("可达未知块数量", float(self.metrics.get("accessible_block_count", 0.0))),
                _safe_metric_text("可达未知未知面积", float(self.metrics.get("total_accessible_unknown_area", 0.0))),
                _safe_metric_text("前沿簇总数", float(self.metrics.get("total_frontier_cluster_count", 0.0))),
                _safe_metric_text("平均未知块面积", float(self.metrics.get("mean_block_area", 0.0))),
                _safe_metric_text("局部前沿覆盖率", float(self.metrics.get("local_frontier_coverage", 0.0))),
                _safe_metric_text("局部前沿块面积均值", float(self.metrics.get("local_frontier_block_area_mean", 0.0))),
            ]
        )
        ax.text(
            0.01,
            0.99,
            semantic_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffffff2", edgecolor="#999999"),
        )
        self._format_axis(ax, belief_map.shape, title="共享语义可视化（Shared Semantic Visualization）")

    def _draw_local_panel(self, ax) -> None:
        assert self.local_snap is not None

        ax.clear()
        if not self.config.show_local_observation:
            ax.text(0.5, 0.5, "局部观测面板已隐藏", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        ax.imshow(self.local_snap, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin="upper", interpolation="nearest")
        center_r, center_c = int(self.sensor.center_state[0]), int(self.sensor.center_state[1])
        ax.add_patch(
            Circle(
                (float(center_c), float(center_r)),
                radius=float(self.sensor.scan_r) + 0.15,
                fill=False,
                edgecolor="#0f4c5c",
                linewidth=1.6,
                linestyle="--",
                alpha=0.85,
                zorder=4,
            )
        )
        ax.scatter(
            [center_c],
            [center_r],
            marker="o",
            s=95,
            c="#f2542d",
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )

        valid_set = set(self.valid_action_indices)
        for key in MOVE_KEY_ORDER:
            action_idx = MOVE_KEY_TO_ACTION[key]
            dr, dc = ACTIONS_8[action_idx]
            text_color = "#2a9d8f" if action_idx in valid_set else "#8d99ae"
            ax.text(
                float(center_c + dc),
                float(center_r + dr),
                key.upper(),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
                zorder=6,
            )

        visible_local = int(np.count_nonzero(self.local_snap != INVISIBLE))
        local_text = "\n".join(
            [
                f"扫描半径：{int(self.sensor.scan_r)}",
                f"可见栅格数：{visible_local}/{int(self.sensor.theoretical_visible_cell_count)}",
                "智能体周围动作按键",
            ]
        )
        ax.text(
            0.01,
            0.99,
            local_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffffff2", edgecolor="#999999"),
        )
        self._format_axis(ax, self.local_snap.shape, title="局部观测 / 雷达足迹（Local Observation / Radar Footprint）")

    def _update_suptitle(self) -> None:
        assert self.figure is not None
        assert self.cum_map is not None
        assert self.agent_state is not None
        title = (
            f"实时交互式语义探索演示 | 步数={self._demo_step()} "
            f"覆盖率={float(self.cum_map.coverage_rate):.1%} "
            f"合法动作数={len(self.valid_action_indices)} "
            f"可达未知块数={int(self.metrics.get('accessible_block_count', 0.0))} "
            f"前沿簇数={int(self.entry_count)} "
            f"智能体=({int(self.agent_state[0])}, {int(self.agent_state[1])})"
        )
        if self.status_message:
            title += f" | {self.status_message}"
        self.figure.suptitle(
            title,
            fontsize=12,
            color="#a61e4d" if self.status_is_error else "#1f2933",
        )

    def render(self) -> None:
        if self.figure is None:
            return
        self._draw_true_map_panel(self.axes["true"])
        self._draw_belief_panel(self.axes["belief"])
        self._draw_semantic_panel(self.axes["semantic"])
        self._draw_local_panel(self.axes["local"])
        self._update_suptitle()
        self.figure.canvas.draw_idle()

    def run(self) -> None:
        self._setup_figure()
        self.render()
        plt.show()


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="实时交互式语义探索演示 demo。")
    parser.add_argument("--rows", type=int, default=40, help="地图行数。默认：40。")
    parser.add_argument("--cols", type=int, default=60, help="地图列数。默认：60。")
    parser.add_argument("--obstacle-ratio", type=float, default=0.20, help="障碍比例。默认：0.20。")
    parser.add_argument("--obs-size", type=int, default=6, help="随机障碍块尺度先验。默认：6。")
    parser.add_argument("--scan-radius", type=int, default=10, help="雷达扫描半径。默认：10。")
    parser.add_argument("--seed", type=int, default=None, help="可选随机种子。")
    parser.add_argument(
        "--show-true-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示或隐藏真值地图面板。",
    )
    parser.add_argument(
        "--show-semantics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启动时启用或关闭语义叠加显示。",
    )
    parser.add_argument(
        "--show-local-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示或隐藏局部观测面板。",
    )
    parser.add_argument(
        "--show-help-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启动时在终端打印按键帮助。",
    )
    args = parser.parse_args()
    return DemoConfig(
        rows=int(args.rows),
        cols=int(args.cols),
        obstacle_ratio=float(args.obstacle_ratio),
        obs_size=int(args.obs_size),
        scan_radius=int(args.scan_radius),
        seed=None if args.seed is None else int(args.seed),
        show_true_map=bool(args.show_true_map),
        show_semantics=bool(args.show_semantics),
        show_local_observation=bool(args.show_local_observation),
        show_help_on_start=bool(args.show_help_on_start),
    )


def main() -> None:
    config = parse_args()
    if config.show_help_on_start:
        print_controls()
    demo = InteractiveSemanticDemo(config)
    demo.run()


if __name__ == "__main__":
    main()
