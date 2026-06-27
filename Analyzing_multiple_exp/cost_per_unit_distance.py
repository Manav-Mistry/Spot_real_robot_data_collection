from cost_of_transport import compute_power, find_distance_covered, find_ave_velocity, BASELINE_DISTANCE, DISTANCE_PER_LOOP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# BASELINE_INCLINE_FLAT =  {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8, "loop": 3},

FILES_INCLINE_FLAT_NPA = [
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate NPA", "distribution": "Front", "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA", "distribution": "Center", "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA", "distribution": "Rear", "loop": 1},
]

# FILES_INCLINE_FLAT_PA = [
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA", "distribution": "Front", "loop": 3},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA", "distribution": "Center", "loop": 3},
# ]
# ----------------------

# BASELINE_INCLINE_CONTINUOUS = {}

FILES_INCLINE_CONTINUOUS_NPA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_continuous_May_20/front_crate_8kg_NPA/incline_continuous_front_crate_8kg_NPA_joints_20260520_121843.csv", "exp_name": "Front Crate NPA", "distribution": "Front",  "loop": 2},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_continuous_May_20/center_crate_8kg_NPA/incline_continuous_center_crate_8kg_NPA_joints_20260520_123445.csv", "exp_name": "Center Crate NPA", "distribution": "Center", "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_continuous_May_20/rear_crate_8kg_NPA/incline_continuous_rear_crate_8kg_NPA_joints_20260529_125013.csv", "exp_name": "Rear Crate NPA", "distribution": "Rear", "loop": 3},

]

FILES_INCLINE_CONTINUOUS_PA = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_continuous_May_20/front_crate_8kg_PA/incline_continuous_front_crate_8kg_PA_joints_20260520_122401.csv", "exp_name": "Front Crate PA", "distribution": "Front",  "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_continuous_May_20/center_crate_8kg_PA/incline_continuous_center_crate_8kg_PA_joints_20260520_125831.csv", "exp_name": "Center Crate PA", "distribution": "Center", "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_continuous_May_20/rear_crate_8kg_PA/incline_continuous_rear_crate_8kg_PA_joints_20260525_151600.csv", "exp_name": "Rear Crate PA", "distribution": "Rear", "loop": 3},

]

DIST_ORDER = ["Front", "Center", "Rear"]

def plot_distribution_vs_terrain(flat_npa, continuous_npa):
    flat_vals = {e["distribution"]: e["energy_per_unit_distance"] for e in flat_npa}
    cont_vals = {e["distribution"]: e["energy_per_unit_distance"] for e in continuous_npa}

    x = np.arange(len(DIST_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, [flat_vals[d] for d in DIST_ORDER], width, color="steelblue", label="Incline Flat NPA")
    ax.bar(x + width/2, [cont_vals[d] for d in DIST_ORDER], width, color="tomato",   label="Incline Continuous NPA")

    ax.set_xticks(x)
    ax.set_xticklabels(DIST_ORDER)
    ax.set_xlabel("Load Distribution")
    ax.set_ylabel("Energy per Unit Distance (J/m)")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("saved_plots/energy_per_distance_vs_terrain.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_distribution_vs_control_mode(continuous_npa, continuous_pa):
    npa_vals = {e["distribution"]: e["energy_per_unit_distance"] for e in continuous_npa}
    pa_vals  = {e["distribution"]: e["energy_per_unit_distance"] for e in continuous_pa}

    x = np.arange(len(DIST_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, [npa_vals[d] for d in DIST_ORDER], width, color="steelblue", label="Not Payload Aware (NPA)")
    ax.bar(x + width/2, [pa_vals[d]  for d in DIST_ORDER], width, color="tomato",   label="Payload Aware (PA)")

    ax.set_xticks(x)
    ax.set_xticklabels(DIST_ORDER)
    ax.set_xlabel("Load Distribution")
    ax.set_ylabel("Energy per Unit Distance (J/m)")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("saved_plots/energy_per_distance_vs_control_mode.png", dpi=300, bbox_inches="tight")
    plt.show()


# metric: total_power / distance
if __name__ == "__main__":
    incline_flat_npa = []
    # incline_flat_pa = []

    incline_continuous_npa = []
    incline_continuous_pa = []

    # for incline flat npa
    for file in FILES_INCLINE_FLAT_NPA:
        df = pd.read_csv(file["data_file"])
         
        total_distance = file["loop"] * DISTANCE_PER_LOOP
        
        total_energy = compute_power(df)

        energy_per_unit_distance = total_energy / (total_distance)

        incline_flat_npa.append({
            "distribution": file["distribution"],
            "energy_per_unit_distance": energy_per_unit_distance,
        })

    # for incline continuous npa
    for file in FILES_INCLINE_CONTINUOUS_NPA:
        df = pd.read_csv(file["data_file"])
         
        total_distance = file["loop"] * DISTANCE_PER_LOOP
        
        total_energy = compute_power(df)

        print(f"Total energy consumption {file['exp_name']}: {total_energy}")

        energy_per_unit_distance = total_energy / (total_distance)

        incline_continuous_npa.append({
            "distribution": file["distribution"],
            "energy_per_unit_distance": energy_per_unit_distance,
        })

    # for incline continuous pa
    for file in FILES_INCLINE_CONTINUOUS_PA:
        df = pd.read_csv(file["data_file"])
         
        total_distance = file["loop"] * DISTANCE_PER_LOOP
        
        total_energy = compute_power(df)

        energy_per_unit_distance = total_energy / (total_distance)

        incline_continuous_pa.append({
            "distribution": file["distribution"],
            "energy_per_unit_distance": energy_per_unit_distance,
        })

    plot_distribution_vs_terrain(incline_flat_npa, incline_continuous_npa)
    plot_distribution_vs_control_mode(incline_continuous_npa, incline_continuous_pa)


