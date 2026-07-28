from __future__ import annotations

"""Shared paper-figure style contract for Matplotlib and Visio exporters."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D

from env.grid_topology import ACTIONS_8


STYLE_CONTRACT_PATH = Path(__file__).with_name("paper_figure_style.json")


@dataclass(frozen=True, slots=True)
class PaperFigureStyle:
    version: str
    contract_path: Path
    occupancy_palette: Mapping[str, str]
    robot_palette: Mapping[str, str]
    robot_geometry_cell_relative: Mapping[str, float]
    fig1_action_palette: Mapping[str, str]
    radar_palette: Mapping[str, str]
    rendering: Mapping[str, float]


def _require_hex_color(value: object, field_name: str) -> str:
    text = str(value).upper()
    if len(text) != 7 or not text.startswith("#"):
        raise ValueError(f"{field_name} must be a six-digit RGB hex color")
    try:
        int(text[1:], 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a six-digit RGB hex color") from exc
    return text


def load_paper_figure_style(path: Path | str = STYLE_CONTRACT_PATH) -> PaperFigureStyle:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    required_sections = (
        "occupancy_palette",
        "robot_palette",
        "robot_geometry_cell_relative",
        "fig1_action_palette",
        "radar_palette",
        "rendering",
    )
    for section in required_sections:
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"style contract is missing mapping section '{section}'")

    occupancy = {
        key: _require_hex_color(payload["occupancy_palette"][key], f"occupancy_palette.{key}")
        for key in ("unknown", "free", "obstacle")
    }
    robot = {
        key: _require_hex_color(payload["robot_palette"][key], f"robot_palette.{key}")
        for key in ("body", "body_edge", "wheels", "radar", "heading")
    }
    fig1_actions = {
        key: _require_hex_color(payload["fig1_action_palette"][key], f"fig1_action_palette.{key}")
        for key in ("legal", "illegal", "selected")
    }
    radar = {
        key: _require_hex_color(payload["radar_palette"][key], f"radar_palette.{key}")
        for key in ("ray", "nominal_boundary", "grid_line", "action")
    }
    geometry_keys = (
        "body_width_cells",
        "body_length_cells",
        "wheel_width_cells",
        "wheel_length_cells",
        "wheel_offset_x_cells",
        "wheel_offset_y_cells",
        "radar_radius_cells",
        "heading_start_cells",
        "heading_length_cells",
        "envelope_target_cells",
    )
    geometry = {key: float(payload["robot_geometry_cell_relative"][key]) for key in geometry_keys}
    rendering = {key: float(value) for key, value in payload["rendering"].items()}
    style = PaperFigureStyle(
        version=str(payload.get("version", "")),
        contract_path=contract_path,
        occupancy_palette=occupancy,
        robot_palette=robot,
        robot_geometry_cell_relative=geometry,
        fig1_action_palette=fig1_actions,
        radar_palette=radar,
        rendering=rendering,
    )
    validate_paper_figure_style(style)
    return style


def robot_envelope_diameter_cells(style: PaperFigureStyle) -> float:
    geometry = style.robot_geometry_cell_relative
    body_radius = math.hypot(
        float(geometry["body_width_cells"]) / 2.0,
        float(geometry["body_length_cells"]) / 2.0,
    )
    wheel_radius = math.hypot(
        float(geometry["wheel_offset_x_cells"]) + float(geometry["wheel_width_cells"]) / 2.0,
        float(geometry["wheel_offset_y_cells"]) + float(geometry["wheel_length_cells"]) / 2.0,
    )
    radar_radius = float(geometry["radar_radius_cells"])
    heading_radius = float(geometry["heading_start_cells"]) + float(geometry["heading_length_cells"])
    return 2.0 * max(body_radius, wheel_radius, radar_radius, heading_radius)


def validate_paper_figure_style(style: PaperFigureStyle) -> None:
    if not style.version:
        raise ValueError("style contract version must not be empty")
    for key, value in style.robot_geometry_cell_relative.items():
        if float(value) <= 0.0:
            raise ValueError(f"robot geometry value '{key}' must be positive")
    envelope = robot_envelope_diameter_cells(style)
    target = float(style.robot_geometry_cell_relative["envelope_target_cells"])
    if envelope > target + 1e-9:
        raise ValueError(f"robot envelope {envelope:.4f} cells exceeds target {target:.4f}")
    normal = float(style.rendering["normal_action_linewidth_pt"])
    selected = float(style.rendering["selected_action_linewidth_pt"])
    if selected < 1.6 * normal:
        raise ValueError("selected action linewidth must be at least 1.6x the normal linewidth")


def occupancy_colormap(style: PaperFigureStyle) -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(
        (
            style.occupancy_palette["unknown"],
            style.occupancy_palette["free"],
            style.occupancy_palette["obstacle"],
        )
    )
    return cmap, BoundaryNorm((-1.5, -0.5, 0.5, 1.5), cmap.N)


def heading_angle_deg(action_index: int) -> float:
    delta_row, delta_col = ACTIONS_8[int(action_index)]
    return float(math.degrees(math.atan2(float(delta_col), -float(delta_row))))


def draw_topdown_robot(
    ax,
    *,
    row: float,
    col: float,
    heading_action: int,
    style: PaperFigureStyle,
    alpha: float = 1.0,
    zorder: int = 12,
) -> tuple[object, ...]:
    """Draw the single shared, cell-relative robot glyph."""

    geometry = style.robot_geometry_cell_relative
    palette = style.robot_palette
    angle = heading_angle_deg(heading_action)
    transform = Affine2D().rotate_deg_around(float(col), float(row), angle) + ax.transData
    parts: list[object] = []

    body_width = float(geometry["body_width_cells"])
    body_length = float(geometry["body_length_cells"])
    body = FancyBboxPatch(
        (float(col) - body_width / 2.0, float(row) - body_length / 2.0),
        body_width,
        body_length,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        facecolor=palette["body"],
        edgecolor=palette["body_edge"],
        linewidth=float(style.rendering["body_edge_linewidth_pt"]),
        alpha=float(alpha),
        transform=transform,
        zorder=zorder,
    )
    ax.add_patch(body)
    parts.append(body)

    wheel_width = float(geometry["wheel_width_cells"])
    wheel_length = float(geometry["wheel_length_cells"])
    wheel_offset_x = float(geometry["wheel_offset_x_cells"])
    wheel_offset_y = float(geometry["wheel_offset_y_cells"])
    for offset_x in (-wheel_offset_x, wheel_offset_x):
        for offset_y in (-wheel_offset_y, wheel_offset_y):
            wheel = Ellipse(
                (float(col) + offset_x, float(row) + offset_y),
                width=wheel_width,
                height=wheel_length,
                facecolor=palette["wheels"],
                edgecolor=palette["wheels"],
                linewidth=float(style.rendering["wheel_edge_linewidth_pt"]),
                alpha=float(alpha),
                transform=transform,
                zorder=zorder - 1,
            )
            ax.add_patch(wheel)
            parts.append(wheel)

    radar = Circle(
        (float(col), float(row)),
        radius=float(geometry["radar_radius_cells"]),
        facecolor=palette["radar"],
        edgecolor=palette["body_edge"],
        linewidth=float(style.rendering["radar_edge_linewidth_pt"]),
        alpha=float(alpha),
        transform=transform,
        zorder=zorder + 1,
    )
    ax.add_patch(radar)
    parts.append(radar)

    heading_start = float(geometry["heading_start_cells"])
    heading_end = heading_start + float(geometry["heading_length_cells"])
    heading = FancyArrowPatch(
        posA=(float(col), float(row) - heading_start),
        posB=(float(col), float(row) - heading_end),
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=float(style.rendering["heading_linewidth_pt"]),
        color=palette["heading"],
        alpha=float(alpha),
        transform=transform,
        zorder=zorder + 2,
    )
    ax.add_patch(heading)
    parts.append(heading)
    return tuple(parts)
