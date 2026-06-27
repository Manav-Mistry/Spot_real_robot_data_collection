import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# All NPA
FILES = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline_incline_flat"},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA"},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA"},

    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA"},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_NPA/incline_flat_stack_center_8kg_NPA_joints_20260428_132539.csv", "exp_name": "Stack Center NPA"},

    # {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent Center NPA"},
]

# All PA
# FILES = [
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline_incline_flat"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA", "mass": 33.8+11.6, "distribution": "Front", "control_mode": "PA", "position": "front", "loop": 3},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8+11.6, "distribution": "Center", "control_mode": "PA", "position": "center", "loop": 3},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_8kg_PA/incline_flat_rear_crate_8kg_PA_joints_20260427_143543.csv", "exp_name": "Rear Crate PA", "mass": 33.8+11.6, "distribution": "Rear", "control_mode": "PA" , "position": "rear", "loop": 3},
# ]


LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]

JOINT_POS_COLS = [f"{leg}.{joint}_position" for leg in LEGS for joint in JOINTS]


def find_joint_motion_range(df):
    rom = {}
    for leg in LEGS:
        for joint in JOINTS:
            col = f"{leg}.{joint}_position"
            positions = df[col].to_numpy()
            rom[f"{leg}.{joint}"] = {
                "min": float(np.min(positions)),
                "max": float(np.max(positions)),
                "range": float(np.max(positions) - np.min(positions)),
            }
    return rom


def plot_joint_motion_range(results):
    exp_names = [r["name"] for r in results]
    n_exps = len(exp_names)
    y = np.arange(n_exps)

    # 4 rows = legs, 3 cols = joints
    fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharey=True)

    for i, leg in enumerate(LEGS):
        for j, joint in enumerate(JOINTS):
            ax = axes[i][j]
            key = f"{leg}.{joint}"

            mins   = np.array([r[key]["min"]   for r in results])
            maxs   = np.array([r[key]["max"]   for r in results])

            # horizontal range lines
            ax.hlines(y, mins, maxs, colors="steelblue", linewidth=2)
            # dots at both ends
            ax.scatter(mins, y, color="steelblue", zorder=3, s=40)
            ax.scatter(maxs, y, color="steelblue", zorder=3, s=40)

            ax.set_title(f"{leg.upper()}.{joint}", fontsize=10)
            ax.set_yticks(y)
            ax.set_yticklabels(exp_names, fontsize=7)
            # ax.set_xlabel("Position (rad)", fontsize=8)
            ax.grid(True, axis="x")

    fig.suptitle("Joint Motion Range across Experiments - PA", fontsize=13)
    plt.tight_layout()
    plt.savefig("saved_plots/joint_motion_range.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_joint_motion_range_bar(results):
    exp_names = [r["name"] for r in results]
    n_exps = len(exp_names)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_exps)]

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharey=False)

    for i, leg in enumerate(LEGS):
        for j, joint in enumerate(JOINTS):
            ax = axes[i][j]
            key = f"{leg}.{joint}"

            ranges = [r[key]["range"] for r in results]

            for k, (val, color) in enumerate(zip(ranges, colors)):
                ax.bar(k, val, color=color, alpha=0.85, edgecolor="white")

            ax.set_title(f"{leg.upper()}.{joint}", fontsize=10)
            ax.set_xticks([])
            ax.set_ylim(0, 2)
            ax.grid(True, axis="y")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.85) for k in range(n_exps)]
    fig.legend(handles, exp_names, loc="lower center", fontsize=12, ncol = 4,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    fig.supylabel("Range (rad)")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("saved_plots/joint_motion_range_bar_NPA.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results = []

    for file in FILES:
        df = pd.read_csv(file["data_file"])

        all_joint_motion_range = find_joint_motion_range(df)

        entry = {"name": file["exp_name"]}
        for leg in LEGS:
            for joint in JOINTS:
                key = f"{leg}.{joint}"
                entry[key] = all_joint_motion_range[key]

        results.append(entry)

    # plot_joint_motion_range(results)
    plot_joint_motion_range_bar(results)
