from __future__ import annotations

from pathlib import Path

import numpy as np

from env.grid_topology import OBSTACLE


def _warn(message: str) -> None:
    print(f"[trajectory] warning: {message}")


def _format_background(true_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(true_grid, dtype=np.int8)
    return np.where(grid == OBSTACLE, 0.15, 1.0).astype(np.float32)


def save_episode_trajectory_plots(
    run_dir: Path,
    episodes: list[dict],
    prefix: str,
    max_episodes: int = 1,
) -> list[Path]:
    if len(episodes) <= 0 or max_episodes <= 0:
        return []

    trajectories_dir = Path(run_dir) / "trajectories"
    try:
        trajectories_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _warn(f"failed to create trajectory directory {trajectories_dir}: {exc}")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _warn(f"matplotlib unavailable, skip trajectory plots: {exc}")
        return []

    generated: list[Path] = []
    limit = min(int(max_episodes), len(episodes))
    for ep_idx in range(limit):
        ep = episodes[ep_idx]
        true_grid = ep.get("true_grid")
        trajectory = ep.get("trajectory_positions")
        if true_grid is None or trajectory is None:
            _warn(f"missing trajectory data for {prefix} ep{ep_idx}")
            continue

        if len(trajectory) <= 0:
            _warn(f"empty trajectory for {prefix} ep{ep_idx}")
            continue

        rows = [int(pos[0]) for pos in trajectory]
        cols = [int(pos[1]) for pos in trajectory]
        background = _format_background(np.asarray(true_grid))

        fig = None
        try:
            height, width = background.shape
            fig_w = max(5.0, min(9.0, width / 6.0))
            fig_h = max(5.0, min(9.0, height / 6.0))

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.imshow(background, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            ax.plot(cols, rows, color="tab:blue", linewidth=2.0)
            ax.scatter([cols[0]], [rows[0]], c="tab:green", marker="o", s=45, label="start")
            ax.scatter([cols[-1]], [rows[-1]], c="tab:red", marker="x", s=55, label="end")
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_title(
                f"{prefix} ep{ep_idx} reward={float(ep.get('episode_reward', 0.0)):.3f} "
                f"cov={float(ep.get('final_coverage', 0.0)):.3f}"
            )
            ax.legend(loc="upper right")
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            fig.tight_layout()

            out_path = trajectories_dir / f"{prefix}_ep{ep_idx}_trajectory.png"
            fig.savefig(out_path, dpi=150)
            generated.append(out_path)
        except Exception as exc:
            _warn(f"failed to render {prefix} ep{ep_idx}: {exc}")
        finally:
            if fig is not None:
                plt.close(fig)

    return generated
