import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILES_PA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent_center_13.6kg", "mass": 33.8 + 8. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_PA/incline_flat_12kg_adj_center_PA_joints_20260426_160040.csv", "exp_name": "Adjacent_center_17.6kg", "mass": 33.8 + 12. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_PA/incline_flat_14kg_adj_center_PA_joints_20260426_171228.csv", "exp_name": "Adjacent_center_19.6kg", "mass": 33.8 + 14. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_16kg_PA/incline_flat_center_crate_16kg_PA_joints_20260428_123328.csv", "exp_name": "Adjacent_center_21.6kg", "mass": 33.8 + 16. + 5.6, "loop": 3},
]

# FILES_NPA = [
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3}, 
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent_center_13.6kg", "mass": 33.8 + 8. + 5.6, "loop": 3}, 
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_NPA/incline_flat_12kg_adj_center_NPA_joints_20260426_155231.csv", "exp_name": "Adjacent_center_17.6kg", "mass": 33.8 + 12. + 5.6, "loop": 3}, 
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_NPA/incline_flat_14kg_adj_center_NPA_joints_20260426_171621.csv", "exp_name": "Adjacent_center_19.6kg", "mass": 33.8 + 14. + 5.6, "loop": 3},
#     {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_June_11/adjacent_center_16kg/incline_flat_adjacent_center_16kg_NPA_joints_20260611_161943.csv", "exp_name": "Adjacent_center_21.6kg", "mass": 33.8 + 16 + 5.6, "loop": 3} 
# ]

# FILES = [
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline", "mass": 33.8}, 
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_joints.csv", "exp_name": "Center Crate", "mass": 33.8+ 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_frontcrate_m_8kg/exp1/flat_frontcrate_m_8kg_exp1_joints.csv", "exp_name": "Front Crate", "mass": 33.8+ 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_rearcrate_m_8kg/exp2/flat_rearcrate_m_8kg_exp2_joints.csv", "exp_name": "Rear Crate", "mass": 33.8+ 11.6},
# ]
DT = 333
LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]

TORQUE_COLS = [f"{leg}.{joint}_load" for leg in LEGS for joint in JOINTS]
ANG_VEL_COLS = [f"{leg}.{joint}_velocity" for leg in LEGS for joint in JOINTS]
 
def find_mean(df: pd.DataFrame, signals: list[str]) -> dict[str, dict]:
    return {
        signal: {
            "mean": float(np.mean(np.abs(df[signal]))),
            "std":  float(np.std(np.abs(df[signal]))),
        }
        for signal in signals
    }

def find_mean_for_velocity(df: pd.DataFrame, signals: list[str]) -> dict[str, dict]:
    return {
        signal: {
            "mean": float(np.mean((df[signal]))),
            "std":  float(np.std((df[signal]))),
        }
        for signal in signals
    }


# def plot_mean_signal(results: list[dict], signal: str, y_limits: tuple):
#     exp_names = [r["exp_name"] for r in results]
#     n_exps = len(exp_names)
#     cmap = plt.get_cmap("tab10")
#     colors = [cmap(i) for i in range(n_exps)]
#
#     fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharey=False)
#
#     for i, leg in enumerate(LEGS):
#         for j, joint in enumerate(JOINTS):
#             ax = axes[i][j]
#             col = f"{leg}.{joint}_{signal}"
#
#             means = [r["mean_values"][col]["mean"] for r in results]
#             stds  = [r["mean_values"][col]["std"]  for r in results]
#
#             for k, (mean, std, color) in enumerate(zip(means, stds, colors)):
#                 ax.bar(k, mean, color=color, alpha=0.85, edgecolor="white",
#                        yerr=std, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "black"})
#
#             ax.set_title(f"{leg.upper()}.{joint}", fontsize=12)
#             ax.set_xticks([])
#             ax.grid(True, axis="y", linestyle="-", alpha=0.9)
#             ax.set_ylim(y_limits[0], y_limits[1])
#
#     handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.85) for k in range(n_exps)]
#     fig.legend(handles, exp_names, loc="lower center", fontsize=11, ncol=n_exps,
#                bbox_to_anchor=(0.5, 0.01), frameon=True)
#
#     fig.supylabel(f"{signal}", fontsize=14)
#     # fig.suptitle("Control Mode - PA")
#     plt.tight_layout(rect=[0, 0.06, 1, 1])
#     plt.savefig(f"mean_{signal}_per_joint.png", dpi=300, bbox_inches="tight")
#     plt.show()


def plot_mean_signal_dots(results: list[dict], signal: str, y_limits: tuple):
    exp_names = [r["exp_name"] for r in results]
    n_exps = len(exp_names)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_exps)]

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharey=False)

    for i, leg in enumerate(LEGS):
        for j, joint in enumerate(JOINTS):
            ax = axes[i][j]
            col = f"{leg}.{joint}_{signal}"

            means = [r["mean_values"][col]["mean"] for r in results]
            stds  = [r["mean_values"][col]["std"]  for r in results]

            # ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

            for k, (mean, std, color) in enumerate(zip(means, stds, colors)):
                ax.errorbar(k, mean, yerr=std, fmt="none",
                            ecolor=color, elinewidth=1.5, capsize=5, capthick=1.5, alpha=0.7)
                ax.scatter(k, mean, color=color, s=180, zorder=5, edgecolors="white", linewidths=0.8)

            ax.set_title(f"{leg.upper()}.{joint}", fontsize=12)
            ax.set_xticks([])
            ax.grid(True, axis="y", linestyle="-", alpha=0.95)
            ax.set_ylim(y_limits[0], y_limits[1])

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[k],
                          markersize=10) for k in range(n_exps)]
    fig.legend(handles, exp_names, loc="lower center", fontsize=11, ncol=n_exps,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    fig.supylabel(f"Angular Velocity (rad/s)", fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f"mean_{signal}_per_joint_PA.png", dpi=300, bbox_inches="tight")
    plt.show()


def find_mean_power_comsumption(mean_torques: dict[str, float], mean_velocities: dict[str, float]):
    power = 0
    for leg in LEGS:
        for joint in JOINTS:
            power += mean_torques[f"{leg}.{joint}_load"]["mean"] * mean_velocities[f"{leg}.{joint}_velocity"]["mean"]
    return power


if __name__ == "__main__":
    results_torque = []
    results_velocity = []
    mean_energy_consumption = []

    for file in FILES_PA:
        df = pd.read_csv(file["data_file"])
        mean_values_torque = find_mean_for_velocity(df, TORQUE_COLS)
        mean_values_velocity = find_mean_for_velocity(df, ANG_VEL_COLS)
        results_torque.append({
            "exp_name": file["exp_name"],
            "mean_values": mean_values_torque,
        })
        results_velocity.append({
            "exp_name": file["exp_name"],
            "mean_values": mean_values_velocity,
        })

        mean_energy_consumption.append({
            "exp_name": file["exp_name"],
            "mean_energy": find_mean_power_comsumption(mean_values_torque, mean_values_velocity) * len(df) * (1/DT),
            "exp_length": len(df)
        })

    # print(results_velocity)
    # plot_mean_signal_dots(results_torque, "load", y_limits=(-20, 80))
    plot_mean_signal_dots(results_velocity, "velocity", y_limits=(-5, 5))

   

    





