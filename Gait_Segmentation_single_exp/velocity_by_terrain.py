import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gait_segmentation_by_terrain as gst
from gait_segmentation import segment_gait_cycles

TERRAIN_RANGES = {
    "flat":       [(0, 1.73), (5.79, 8.97), (13.00, 17.33), (21.53, 24.29), (28.44, 33.25), (37.18, 40.50), (44.13, 1e4)],
    "ascending":  [(1.73, 3.92), (8.97, 10.84), (17.33, 19.57), (24.29, 26.25), (33.25, 35.26), (40.5, 42.31)],
    "descending": [(3.92, 5.79), (10.84, 13.00), (19.57, 21.53), (26.25, 28.44), (35.26, 37.18), (42.31, 44.13)],
}
gst.TERRAIN_RANGES = TERRAIN_RANGES

# FILES = [
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "Baseline"},

#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_NPA/incline_flat_stack_center_8kg_NPA_joints_20260428_132539.csv", "exp_name": "Stack Center NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adj Center NPA"},

#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_PA/incline_flat_stack_center_8kg_PA_joints_20260428_131820.csv", "exp_name": "Stack Center PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adj Center PA"},
# ]

# Weight: 8kg, Distributions: Front, Center, Rear, Stack, Adj

# FILES_NPA = [
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "Baseline"},

#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_NPA/incline_flat_stack_center_8kg_NPA_joints_20260428_132539.csv", "exp_name": "Stack Center NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adj Center NPA"},
# ]

# FILES_PA = [
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_PA/incline_flat_stack_center_8kg_PA_joints_20260428_131820.csv", "exp_name": "Stack Center PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adj Center PA"},

# ]


# Distribution : Adjacent Center, Weight: 8, 12, 14, 16 kg
FILES_NPA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_NPA/incline_flat_12kg_adj_center_NPA_joints_20260426_155231.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_NPA/incline_flat_14kg_adj_center_NPA_joints_20260426_171621.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14. + 5.6, "loop": 3}, 
]

FILES_PA = [
    # {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_PA/incline_flat_12kg_adj_center_PA_joints_20260426_160040.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_PA/incline_flat_14kg_adj_center_PA_joints_20260426_171228.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14. + 5.6, "loop": 3}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_16kg_PA/incline_flat_center_crate_16kg_PA_joints_20260428_123328.csv", "exp_name": "Adjacent_center_16kg", "mass": 33.8 + 16. + 5.6, "loop": 3}, 
]

TERRAINS = ["flat", "ascending", "descending"]


def plot_velocity_by_terrain_vs_weight(results_npa: list, results_pa: list) -> None:
    from matplotlib.patches import Patch

    n_npa = len(results_npa)
    n_pa  = len(results_pa)
    pos_npa = list(range(1, n_npa + 1))
    pos_pa  = list(range(n_npa + 2, n_npa + 2 + n_pa))

    names_npa = [r["exp_name"] for r in results_npa]
    names_pa  = [r["exp_name"] for r in results_pa]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, terrain in zip(axes, TERRAINS):
        data_npa = [r[terrain] for r in results_npa]
        data_pa  = [r[terrain] for r in results_pa]

        bp_npa = ax.boxplot(data_npa, positions=pos_npa, patch_artist=True, showfliers=True, medianprops=dict(linewidth=2.5))
        bp_pa  = ax.boxplot(data_pa,  positions=pos_pa,  patch_artist=True, showfliers=True, medianprops=dict(linewidth=2.5))

        for patch in bp_npa["boxes"]:
            patch.set_facecolor("lightblue")
        for patch in bp_pa["boxes"]:
            patch.set_facecolor("lightgreen")

        ax.set_ylabel("Velocity (m/s)")
        ax.set_title(terrain.capitalize())
        ax.grid(True, axis="y")
        ax.set_xlim(0, n_npa + n_pa + 2)
        ax.set_xticks(pos_npa + pos_pa)
        ax.set_xticklabels(names_npa + names_pa, rotation=30, ha="right")

    axes[0].legend(handles=[Patch(facecolor="lightblue", label="NPA"),
                             Patch(facecolor="lightgreen", label="PA")])
    fig.supylabel("Mean Speed per Gait Cycle (m/s)")
    plt.tight_layout()
    plt.savefig("velocity_by_terrain_vs_weights.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results_npa = []
    results_pa = []

    for file in FILES_NPA:
        df = pd.read_csv(file["data_file"])

        cum_dist   = gst.compute_cumulative_distance(df)
        segments   = segment_gait_cycles(df)
        by_terrain = gst.classify_segments_by_terrain(segments, cum_dist)

        entry = {"exp_name": file["exp_name"]}
        for terrain in TERRAINS:
            segs = by_terrain.get(terrain, [])
            entry[terrain] = gst.find_vel_per_gait(df, segs) if segs else np.array([])
            print(f"[{file['exp_name']}] {terrain}: {len(segs)} gait cycles")

        results_npa.append(entry)
    
    for file in FILES_PA:
        df = pd.read_csv(file["data_file"])

        cum_dist   = gst.compute_cumulative_distance(df)
        segments   = segment_gait_cycles(df)
        by_terrain = gst.classify_segments_by_terrain(segments, cum_dist)

        entry = {"exp_name": file["exp_name"]}
        for terrain in TERRAINS:
            segs = by_terrain.get(terrain, [])
            entry[terrain] = gst.find_vel_per_gait(df, segs) if segs else np.array([])
            print(f"[{file['exp_name']}] {terrain}: {len(segs)} gait cycles")

        results_pa.append(entry)

    plot_velocity_by_terrain_vs_weight(results_npa, results_pa)
