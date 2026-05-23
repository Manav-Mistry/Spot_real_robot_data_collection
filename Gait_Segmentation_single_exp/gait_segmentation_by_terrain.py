import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gait_segmentation import segment_gait_cycles, map_phase, resample_cycles

# FILE = "/home/nerve/Desktop/data_collected/incline_crossing_May_13/stack_front_8kg_NPA/incline_crossing_stack_front_8kg_NPA_joints_20260513_170453.csv"
FILE = "/home/nerve/Desktop/data_collected/flat_Mar_20/front_rear/single_tier_front_rear_8kg_NPA_loop3_joints_20260320_143151.csv"
# terrain segments as cumulative distance ranges (start_m, end_m)
# for incline terrain
# TERRAIN_RANGES = {
#     "flat":       [(0, 1.73), (5.79, 8.97), (13.00, 17.33), (21.53, 24.29), (28.44, 33.25), (37.18, 40.50), (44.13, 1e4)],
#     "ascending":  [(1.73, 3.92), (8.97, 10.84), (17.33, 19.57), (24.29, 26.25), (33.25, 35.26), (40.5, 42.31)],
#     "descending": [(3.92, 5.79), (10.84, 13.00), (19.57, 21.53), (26.25, 28.44), (35.26, 37.18), (42.31, 44.13)],
# }

# for flat, flat terrain
TERRAIN_RANGES = {
    "flat": [(0, 1e4)]
}

def compute_cumulative_distance(df: pd.DataFrame) -> np.ndarray:
    x = df["vision_tform_body_pos_x"].values
    y = df["vision_tform_body_pos_y"].values
    z = df["vision_tform_body_pos_z"].values
    step_dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    return np.concatenate([[0], np.cumsum(step_dist)])


def classify_terrain(dist: float) -> str | None:
    for terrain, ranges in TERRAIN_RANGES.items():
        for lo, hi in ranges:
            if lo <= dist < hi:
                return terrain
    return None


def classify_segments_by_terrain(
    segments: list[tuple[int, int]],
    cum_dist: np.ndarray,
) -> dict[str, list[tuple[int, int]]]:
    by_terrain: dict[str, list[tuple[int, int]]] = {t: [] for t in TERRAIN_RANGES}

    for s, e in segments:
        mid = (s + e) // 2
        terrain = classify_terrain(cum_dist[mid])
        if terrain is not None:
            by_terrain[terrain].append((s, e))

    return by_terrain


def plot_mean_std_per_phase_by_terrain(resampled: dict[str, np.ndarray], terrain_label: str, n_points: int = 100) -> None:
    phase   = np.linspace(0, 1, n_points)
    columns = list(resampled.keys())
    n_cols  = len(columns)

    _, axes = plt.subplots(n_cols, 1, figsize=(10, 4 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        matrix = resampled[col]
        mean   = matrix.mean(axis=0)
        std    = matrix.std(axis=0)

        ax.plot(phase, mean, linewidth=2, label="Mean")
        ax.fill_between(phase, mean - std, mean + std, alpha=0.25, label="1 standard deviation")
        ax.set_ylabel(col)
        ax.legend(loc="upper right")
        ax.grid(True)

    axes[0].set_title(f"Gait phase — {terrain_label}")
    axes[-1].set_xlabel("Gait phase")
    plt.tight_layout()
    plt.savefig(f"joint_torque_phase_{terrain_label}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_torque_RMS_per_terrain(by_terrain: dict[str, list[tuple[int, int]]], column: str, df: pd.DataFrame, n_points: int = 100) -> None:
    phase = np.linspace(0, 1, n_points)
    rms_per_terrain: dict[str, np.ndarray] = {}

    for terrain, terrain_segments in by_terrain.items():
        if not terrain_segments:
            continue
        matrix = resample_cycles(df, terrain_segments, [column], n_points)[column]
        rms_per_terrain[terrain] = np.sqrt(np.mean(matrix ** 2, axis=0))

    _, ax = plt.subplots(figsize=(10, 4))
    for terrain, rms in rms_per_terrain.items():
        ax.plot(phase, rms, linewidth=2, label=terrain)

    ax.set_xlabel("Gait phase")
    ax.set_ylabel(column)
    ax.set_title(f"RMS torque per terrain - {column}")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"rms_per_terrain_{column}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_peak_torque_per_terrain(by_terrain: dict[str, list[tuple[int, int]]], column: str, df: pd.DataFrame, n_points: int = 100) -> None:
    phase = np.linspace(0, 1, n_points)
    peak_per_terrain: dict[str, np.ndarray] = {}

    for terrain, terrain_segments in by_terrain.items():
        if not terrain_segments:
            continue
        matrix = resample_cycles(df, terrain_segments, [column], n_points)[column]
        peak_per_terrain[terrain] = np.max(matrix, axis=0)

    _, ax = plt.subplots(figsize=(10, 4))
    for terrain, rms in peak_per_terrain.items():
        ax.plot(phase, rms, linewidth=2, label=terrain)
    
    ax.set_xlabel("Gait phase")
    ax.set_ylabel(column)
    ax.set_title(f"Peak torque per terrain - {column}")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"peak_per_terrain_{column}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_front_rear_effort_ratio(by_terrain: dict[str, list[tuple[int, int]]], front_joints: list[str], rear_joints: list[str], df: pd.DataFrame, n_points: int = 100) -> None:
    phase = np.linspace(0, 1, n_points)
    load_ratio: dict[str, np.ndarray] = {}

    for terrain, terrain_segments in by_terrain.items():
        if not terrain_segments:                                                 # fix: skip empty terrains
            continue

        matrix_f = resample_cycles(df, terrain_segments, front_joints, n_points)
        L_f = sum(np.sqrt(np.mean(matrix_f[j] ** 2, axis=0)) for j in front_joints)

        matrix_r = resample_cycles(df, terrain_segments, rear_joints, n_points)
        L_r = sum(np.sqrt(np.mean(matrix_r[j] ** 2, axis=0)) for j in rear_joints)

        load_ratio[terrain] = (L_f - L_r) / (L_f + L_r)

    _, ax = plt.subplots(figsize=(10, 4))
    for terrain, ratio in load_ratio.items():
        ax.plot(phase, ratio, linewidth=2, label=terrain)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Gait phase")
    ax.set_ylabel("Effort Ratio")
    ax.set_title("Front rear effort ratio - Front Rear Distribution")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("front_rear_effort_ratio_front_rear_dist.png", dpi=300, bbox_inches='tight')
    plt.show()





if __name__ == "__main__":
    df = pd.read_csv(FILE)

    cum_dist = compute_cumulative_distance(df)
    segments = segment_gait_cycles(df)
    df = map_phase(df, segments)

    columns    = ["fl.hx_load", "fl.hy_load", "fl.kn_load"]
    by_terrain = classify_segments_by_terrain(segments, cum_dist)

    # --- three joint torques per leg in the same plot
    # for terrain, terrain_segments in by_terrain.items():
    #     print(f"{terrain}: {len(terrain_segments)} gait cycles")
    #     if not terrain_segments:
    #         continue
    #     resampled = resample_cycles(df, terrain_segments, columns=columns, n_points=100)
    #     plot_mean_std_per_phase_by_terrain(resampled, terrain_label=terrain)

    # --- comparison RMS torque for a single joint on different terrains
    # plot_torque_RMS_per_terrain(by_terrain, "fl.hy_load", df, n_points=100)

    # --- plot peak torque for a signal joint on different terrains
    # plot_peak_torque_per_terrain(by_terrain, "fl.hx_load", df, n_points=100)
    
    # --- front rear effort ratio ----------------------------------
    front_joints = ["fl.hx_load", "fl.hy_load", "fl.kn_load", "fr.hx_load", "fr.hy_load", "fr.kn_load"]
    rear_joints = ["hl.hx_load", "hl.hy_load", "hl.kn_load", "hr.hx_load", "hr.hy_load", "hr.kn_load"]

    plot_front_rear_effort_ratio(by_terrain, front_joints, rear_joints, df, n_points=100)