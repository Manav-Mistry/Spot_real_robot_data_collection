import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILES_PA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adj_center_13.6kg", "mass": 33.8 + 8. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_PA/incline_flat_12kg_adj_center_PA_joints_20260426_160040.csv", "exp_name": "Adj_center_17.6kg", "mass": 33.8 + 12. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_PA/incline_flat_14kg_adj_center_PA_joints_20260426_171228.csv", "exp_name": "Adj_center_19.6kg", "mass": 33.8 + 14. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_16kg_PA/incline_flat_center_crate_16kg_PA_joints_20260428_123328.csv", "exp_name": "Adj_center_21.6kg", "mass": 33.8 + 16. + 5.6, "loop": 3},
]

LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]

TORQUE_COLS = [f"{leg}.{joint}_load" for leg in LEGS for joint in JOINTS]

ANG_VEL_COLS = [f"{leg}.{joint}_velocity" for leg in LEGS for joint in JOINTS]



def find_max_limits(df: pd.DataFrame, signals: list[str]) -> dict[str, float]:
    return {signal: float(np.max(np.abs(df[signal]))) for signal in signals}


def plot_max_signal(results: list[dict], signal: str, y_limits: tuple):
    exp_names = [r["exp_name"] for r in results]
    n_exps = len(exp_names)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_exps)]

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharey=False)

    for i, leg in enumerate(LEGS):
        for j, joint in enumerate(JOINTS):
            ax = axes[i][j]
            col = f"{leg}.{joint}_{signal}"

            values = [r["max_values"][col] for r in results]

            for k, (val, color) in enumerate(zip(values, colors)):
                ax.bar(k, val, color=color, alpha=0.85, edgecolor="white")

            ax.set_title(f"{leg.upper()}.{joint}", fontsize=10)
            ax.set_xticks([])
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)
            ax.set_ylim(y_limits[0], y_limits[1])

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.85) for k in range(n_exps)]
    fig.legend(handles, exp_names, loc="lower center", fontsize=11, ncol=n_exps,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    fig.supylabel(f"Maximum {signal}", fontsize=14)
    # fig.suptitle("Control Mode - PA")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f"max_{signal}_per_joint.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results = []
    for file in FILES_PA:
        df = pd.read_csv(file["data_file"])
        max_values = find_max_limits(df, TORQUE_COLS)
        results.append({
            "exp_name": file["exp_name"],
            "max_values": max_values,
        })

    # plot_max_signal(results, "velocity", y_limits=(0, 15))
    plot_max_signal(results, "load", y_limits=(0, 150))

