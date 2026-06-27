import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vibration_cost import compute_vibration_cost, JOINT_POS_COLS, FS
from cost_of_transport import DISTANCE_PER_LOOP

BASELINE_FILE = "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv"

FILES_NPA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_NPA/incline_flat_12kg_adj_center_NPA_joints_20260426_155231.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_NPA/incline_flat_14kg_adj_center_NPA_joints_20260426_171621.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14. + 5.6, "loop": 3},
]

FILES_PA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_PA/incline_flat_12kg_adj_center_PA_joints_20260426_160040.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_PA/incline_flat_14kg_adj_center_PA_joints_20260426_171228.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14. + 5.6, "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_16kg_PA/incline_flat_center_crate_16kg_PA_joints_20260428_123328.csv", "exp_name": "Adjacent_center_16kg", "mass": 33.8 + 16. + 5.6, "loop": 3},
]


def find_limits(baseline_file):
    df = pd.read_csv(baseline_file)
    limits = []
    for col in JOINT_POS_COLS:
        positions = df[col].to_numpy()
        limits.append({
            "joint_col_name": col,
            "lower_limit": np.percentile(positions, 5),
            "upper_limit": np.percentile(positions, 95),
        })
    return limits


def process_files(files, limits):
    results = []
    for file in files:
        df = pd.read_csv(file["data_file"])

        if "start" in file:
            df = df.iloc[file["start"]: file["end"]]

        cost, _, _ = compute_vibration_cost(df, limits)
        # total_cost = cost_per_sample * len(df)
        # fixed_distance = file["loop"] * DISTANCE_PER_LOOP
        # vib_cost_per_distance = total_cost / fixed_distance

        print(f"[{file['exp_name']}] Vibration cost per unit distance: {cost:.4f}")

        results.append({
            "name": file["exp_name"],
            "payload": file["mass"],
            "vibration_cost": cost / file["loop"],
        })
    return results


if __name__ == "__main__":
    limits = find_limits(BASELINE_FILE)

    results_NPA = process_files(FILES_NPA, limits)
    results_PA  = process_files(FILES_PA,  limits)

    payload_NPA = [r["payload"] - 33.8 for r in results_NPA]
    payload_PA  = [r["payload"] - 33.8 for r in results_PA]

    vib_NPA = [r["vibration_cost"] for r in results_NPA]
    vib_PA  = [r["vibration_cost"] for r in results_PA]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(payload_NPA, vib_NPA, "-o",  color="black", label="Not Payload Aware")
    ax.plot(payload_PA,  vib_PA,  "--o", color="black", label="Payload Aware")

    ax.set_xlabel("Payload Weight (kg)")
    ax.set_ylabel("Vibration Cost (Per trial)")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(payload_PA)

    plt.tight_layout()
    plt.savefig("vibration_cost_vs_weight.png", dpi=300, bbox_inches='tight')
    plt.show()
