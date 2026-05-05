import pandas as pd
import numpy as np

BASELINE_FILE = "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv"

FILES = [
    # All NPA
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline"},

    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA"},

    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA"},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA"},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_NPA/incline_flat_stack_center_8kg_NPA_joints_20260428_132539.csv", "exp_name": "Stack Center NPA"},

    # All PA
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA"},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_PA/incline_flat_stack_center_8kg_PA_joints_20260428_131820.csv", "exp_name": "Stack Center PA"},
]

LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]

JOINT_POS_COLS = [f"{leg}.{joint}_position" for leg in LEGS for joint in JOINTS]


def find_limits():
    df = pd.read_csv(BASELINE_FILE)
    limits = []
    for col in JOINT_POS_COLS:
        positions = df[col].to_numpy()
        limits.append({
            "joint_col_name": col,
            "lower_limit": np.percentile(positions, 5),
            "upper_limit": np.percentile(positions, 95),
        })
    return limits


FS = 333

def compute_vibration_cost(df, limits):
    n_samples = len(df)
    duration_s = n_samples / FS
    total_cost = 0.0
    for entry in limits:
        col   = entry["joint_col_name"]
        lower = entry["lower_limit"]
        upper = entry["upper_limit"]

        positions = df[col].to_numpy()
        below = np.where(positions < lower, np.abs(positions - lower), 0.0)
        above = np.where(positions > upper, np.abs(positions - upper), 0.0)
        total_cost += (below + above).sum()

    return total_cost / n_samples, total_cost / duration_s


if __name__ == "__main__":
    limits = find_limits()
    results = []

    for file in FILES:
        df = pd.read_csv(file["data_file"])

        if "start" in file:
            df = df.iloc[file["start"]: file["end"]]

        cost_per_sample, cost_per_second = compute_vibration_cost(df, limits)
        print(f"[{file['exp_name']}] Vibration Cost: {cost_per_sample:.4f} (per sample) | {cost_per_second:.4f} (per second)")

        results.append({
            "name": file["exp_name"],
            "vibration_cost_per_sample": round(cost_per_sample, 4),
            "vibration_cost_per_second": round(cost_per_second, 4),
        })

    # print(results)
