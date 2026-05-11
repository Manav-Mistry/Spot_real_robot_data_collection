import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from vibration_cost import find_limits, compute_vibration_cost
from cost_of_transport import find_distance_covered, compute_power

BASELINE_FILE = "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv"

FILES_SAND = [
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_joints.csv", "exp_name": "Center Crate NPA", "mass": 11.6 , "terrain": "Flat"}, 
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 11.6, "terrain": "Incline_Flat"}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_crossover_May_9/center_crate_8kg_NPA/incline_crossover_center_crate_8kg_NPA_joints_20260509_163246.csv", "exp_name": "Center Crate NPA", "mass": 11.6, "terrain": "Incline_Crossing"}, 
]

FILES_WATER = [
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Apr_7/water_center/water_center_8kg_joints_20260407_195525.csv", "exp_name": "Center Crate NPA", "mass": 11.6, "terrain": "Flat"}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_27/water_center_8kg_NPA/incline_flat_water_center_crate_8kg_NPA_joints_20260427_120601.csv", "exp_name": "Center Crate NPA", "mass": 11.6, "terrain": "Incline_Flat"}, 
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_crossover_May_11/water_center_crate_8kg_NPA/incline_crossover_center_crate_water_8kg_NPA_joints_20260511_164738.csv", "exp_name": "Center NPA", "mass": 11.6, "terrain": "Incline_Crossing"},
]

LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]

JOINT_POS_COLS = [f"{leg}.{joint}_position" for leg in LEGS for joint in JOINTS]

def plot_vibration_vs_terrain(results_sand, results_water):
    x_axis = ["Flat", "Incline_Flat", "Incline_Crossing"]
    x_order = {"Flat": 0, "Incline_Flat": 1, "Incline_Crossing": 2}

    def _xy(results):
        sorted_results = sorted(results, key=lambda r: x_order.get(r["terrain"]))
        x = [x_order[r["terrain"]] for r in sorted_results]
        y = [r["vibration_cost_per_second"] for r in sorted_results]
        return x, y

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(*_xy(results_sand),  "-o",  color="black", label="Sand")
    ax.plot(*_xy(results_water), "-o", color="blue",  label="Water")

    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_xlabel("Terrain")
    ax.set_ylabel("Vibration Cost Per Second (rad/s)")
    ax.legend()
    ax.grid(True)

    fig.suptitle("")
    plt.tight_layout()
    plt.savefig("vibration_cost_vs_terrain.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_cot_vs_terrain(results_sand, results_water):
    x_axis = ["Flat", "Incline_Flat", "Incline_Crossing"]
    x_order = {"Flat": 0, "Incline_Flat": 1, "Incline_Crossing": 2}

    def _xy(results):
        sorted_results = sorted(results, key=lambda r: x_order.get(r["terrain"]))
        x = [x_order[r["terrain"]] for r in sorted_results]
        y = [r["cot"] for r in sorted_results]
        return x, y

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(*_xy(results_sand),  "-o",  color="black", label="Sand")
    ax.plot(*_xy(results_water), "-o", color="blue",  label="Water")

    ax.set_xticks(range(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_xlabel("Terrain")
    ax.set_ylabel("Cost of Transport")
    ax.legend()
    ax.grid(True)

    fig.suptitle("")
    plt.tight_layout()
    plt.savefig("cot_vs_terrain.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    limits = find_limits()
    results_sand = []
    results_water = []

    # for sand payload
    for file in FILES_SAND:
        df = pd.read_csv(file["data_file"])

        if "start" in file:
            df = df.iloc[file["start"]: file["end"]]

        cost_per_sample, cost_per_second = compute_vibration_cost(df, limits)
        # print(f" sand [{file['exp_name']}] Vibration Cost: {cost_per_sample:.4f} (per sample) | {cost_per_second:.4f} (per second)")

        total_distance= find_distance_covered(df)
        total_energy = compute_power(df)

        cot_energy = total_energy / (file["mass"] * 9.8 * total_distance)
        
        results_sand.append({
            "name": file["exp_name"],
            "vibration_cost_per_sample": round(cost_per_sample, 4),
            "vibration_cost_per_second": round(cost_per_second, 4),
            "cot": round(cot_energy, 4),
            "terrain": file["terrain"]
        })

    # for water payload
    for file in FILES_WATER:
        df = pd.read_csv(file["data_file"])

        if "start" in file:
            df = df.iloc[file["start"]: file["end"]]

        cost_per_sample, cost_per_second = compute_vibration_cost(df, limits)
        # print(f"water [{file['exp_name']}] Vibration Cost: {cost_per_sample:.4f} (per sample) | {cost_per_second:.4f} (per second)")

        total_distance= find_distance_covered(df)
        total_energy = compute_power(df)

        cot_energy = total_energy / (file["mass"] * 9.8 * total_distance)
        
        results_water.append({
            "name": file["exp_name"],
            "vibration_cost_per_sample": round(cost_per_sample, 4),
            "vibration_cost_per_second": round(cost_per_second, 4),
            "cot": round(cot_energy, 4),
            "terrain": file["terrain"]
        })

    
    plot_vibration_vs_terrain(results_sand, results_water)
    plot_cot_vs_terrain(results_sand, results_water)


