import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Different Terrain, Center Distribution
FILES = [
    {"datafile": "/home/nerve/Desktop/data_collected/incline_continuous_May_20/center_crate_8kg_NPA/incline_continuous_center_crate_8kg_NPA_joints_20260520_123445.csv", "exp_name": "Incline, Continuous"},
    {"datafile": "/home/nerve/Desktop/data_collected/incline_crossover_May_9/center_crate_8kg_NPA/incline_crossover_center_crate_8kg_NPA_joints_20260509_163246.csv", "exp_name": "Incline, Crossing"},
    {"datafile": "/home/nerve/Desktop/data_collected/incline_foam_May_18/center_crate_8kg_NPA/incline_foam_center_crate_8kg_NPA_joints_20260518_142503.csv", "exp_name": "Incline, Foam"},
]

# Incline Flat, Front, NPA vs PA
# FILES = [
#     {"datafile": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate, NPA"},
#     {"datafile": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate, PA"},
# ]

LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]

TORQUE_COLS = [f"{leg}.{joint}_load" for joint in JOINTS for leg in LEGS]
ANGULAR_VEL_COLS = [f"{leg}.{joint}_velocity" for joint in JOINTS for leg in LEGS]

def find_rms_velocity_per_joint(df:pd.DataFrame) -> list[float]: 
    result = []
    for joint in ANGULAR_VEL_COLS:
        result.append(np.sqrt(np.mean(df[joint]**2)))
    return result


def find_rms_torques_per_joint(df:pd.DataFrame) -> list[float]: 
    result = []
    for joint in TORQUE_COLS:
        result.append(np.sqrt(np.mean(df[joint]**2)))
    return result

def find_mean_std_torques_per_joint(df: pd.DataFrame) -> list[float]:
    mean_result = []
    std_result = []
    for joint in TORQUE_COLS:
        mean_result.append(np.mean(df[joint]))
        std_result.append(np.std(df[joint]))
    return mean_result, std_result 

def find_peak_torques_per_joint(df:pd.DataFrame):
    result = []
    for joint in TORQUE_COLS:
        result.append(np.max(np.abs(df[joint])))
    return result


JOINT_LABELS = [f"{leg.upper()}.{joint.upper()}" for joint in JOINTS for leg in LEGS]
COLORS = plt.get_cmap("tab10").colors


def plot_rms_velocity(velocity_results: list[dict]):
    n_joints = len(TORQUE_COLS)
    n_exps = len(velocity_results)
    x = np.arange(n_joints)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 3))
    for i, result in enumerate(velocity_results):
        offset = (i - n_exps / 2 + 0.5) * width
        ax.bar(x + offset, result["rms_velocity_per_joint"], width,
               label=result["exp_name"], color=COLORS[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS, rotation=0, ha="center")
    # ax.set_xlabel("Joint")
    ax.set_ylabel("RMS Velocity (rad/s)", fontsize=12)
    # ax.set_title("RMS Torque per Joint across Experiments")
    ax.legend()
    ax.grid(axis="y", linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("rms_ang_vel_per_joint_front_crate_NPA_vs_PA.png", dpi=150)
    plt.show()

def plot_rms_torques(torque_results: list[dict]):
    n_joints = len(TORQUE_COLS)
    n_exps = len(torque_results)
    x = np.arange(n_joints)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 3))
    for i, result in enumerate(torque_results):
        offset = (i - n_exps / 2 + 0.5) * width
        ax.bar(x + offset, result["rms_torques_per_joint"], width,
               label=result["exp_name"], color=COLORS[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS, rotation=0, ha="center")
    # ax.set_xlabel("Joint")
    ax.set_ylabel("RMS Torque (Nm)", fontsize=12)
    # ax.set_title("RMS Torque per Joint across Experiments")
    ax.legend()
    ax.grid(axis="y", linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("rms_ang_vel_per_joint_front_crate_NPA_vs_PA.png", dpi=150)
    plt.show()


def plot_rms_torques(torque_results: list[dict]):
    n_joints = len(TORQUE_COLS)
    n_exps = len(torque_results)
    x = np.arange(n_joints)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 3))
    for i, result in enumerate(torque_results):
        offset = (i - n_exps / 2 + 0.5) * width
        ax.bar(x + offset, result["rms_torques_per_joint"], width,
               label=result["exp_name"], color=COLORS[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS, rotation=0, ha="center")
    # ax.set_xlabel("Joint")
    ax.set_ylabel("RMS Torque (Nm)", fontsize=12)
    # ax.set_title("RMS Torque per Joint across Experiments")
    ax.legend()
    ax.grid(axis="y", linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("rms_ang_vel_per_joint_front_crate_NPA_vs_PA.png", dpi=150)
    plt.show()


def plot_mean_std_torques(torque_results: list[dict]):
    n_joints = len(TORQUE_COLS)
    n_exps = len(torque_results)
    x = np.arange(n_joints)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 3))
    for i, result in enumerate(torque_results):
        offset = (i - n_exps / 2 + 0.5) * width
        ax.bar(x + offset, result["mean_torques_per_joint"], width,
               yerr=result["std_torques_per_joint"], capsize=3,
               label=result["exp_name"], color=COLORS[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS, rotation=0, ha="center")
    # ax.set_xlabel("Joint")
    ax.set_ylabel("Torque (Nm)", fontsize=12)
    # ax.set_title("RMS Torque per Joint across Experiments")
    ax.legend()
    ax.grid(axis="y", linestyle="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mean_std_torques_per_joint_front_terrains.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    results = []

    for file in FILES:
        df = pd.read_csv(file["datafile"])
        rms_torques = find_rms_torques_per_joint(df)
        peak_torques = find_peak_torques_per_joint(df)
        rms_velocity = find_rms_velocity_per_joint(df)
        mean_torques, std_torques = find_mean_std_torques_per_joint(df)

        results.append({
            "exp_name": file["exp_name"],
            "rms_torques_per_joint": rms_torques,
            "peak_torques_per_joint": peak_torques,
            "rms_velocity_per_joint": rms_velocity,
            "mean_torques_per_joint": mean_torques,
            "std_torques_per_joint": std_torques
        })

    # plot_rms_torques(results)
    # plot_peak_torques(results)
    plot_mean_std_torques(results)
    # plot_rms_velocity(results)



