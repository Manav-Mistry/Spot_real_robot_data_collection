import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


FILES = [
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline", "start": 11240, "end": 19700, "mass": 33.8}, 
   
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/front/adj_single_tier_front_8kg_NPA_loop3_joints_20260320_125855.csv", "exp_name": "Adj_front_8kg", "start": 500, "end": 10500, "mass": 33.8+ 13.6},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/center/adj_single_tier_center_8kg_NPA_loop3_joints_20260320_132519.csv", "exp_name": "Adj_center_8kg", "start": 2000, "end": 10700, "mass": 33.8+ 13.6},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_center/double_tier_center_8kg_NPA_loop3_joints_20260320_162856.csv", "exp_name": "Stack_center_8kg", "start": 1500, "end": 10000, "mass": 33.8+ 11.6},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_front/double_tier_front_8kg_NPA_loop3_joints_20260320_151917.csv", "exp_name": "Stack_front_8kg", "start": 2000, "end": 12000, "mass": 33.8+ 11.6},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/front_rear/single_tier_front_rear_8kg_NPA_loop3_joints_20260320_143151.csv", "exp_name": "front_rear_8kg", "start": 1600, "end": 10600, "mass": 33.8+ 12.8},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_joints.csv", "exp_name": "center_crate_8kg", "start": 2000, "end": 10500, "mass": 33.8+ 11.6},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_frontcrate_m_8kg/exp1/flat_frontcrate_m_8kg_exp1_joints.csv", "exp_name": "front_crate_8kg", "start": 1500, "end": 11000, "mass": 33.8+ 11.6},

]

LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]
FS = 333
DT = 1/FS

TORQUE_COLS = [f"{leg}.{joint}_load" for joint in JOINTS for leg in LEGS]
ANG_VEL_COLS = [f"{leg}.{joint}_velocity" for joint in JOINTS for leg in LEGS]

TORQUE_VELOCITY_PAIR = list(zip(TORQUE_COLS, ANG_VEL_COLS))

def compute_power(df):
    power_joint_t =  [ np.abs(df[t] * df[v]) for t, v in TORQUE_VELOCITY_PAIR ]
    power_t = np.sum(power_joint_t) * (DT)
    total_power = np.sum(power_t)
    return total_power


def find_avg_velocity(df):
    forward_velocity_array = df["vel_of_body_in_vision_lin_y"]
    return np.mean(np.abs(forward_velocity_array))

results = []

# --- find cost of transport ---------------------------
for file in FILES:
    df = pd.read_csv(file["data_file"])
    start = file["start"]
    end = file["end"]
    df = df.iloc[start: end]
    power = compute_power(df)
    avg_velocity = find_avg_velocity(df)
    print("avg vel: ", avg_velocity)
    cot = power / file["mass"] * 9.8 * avg_velocity

    results.append({
        "name": file["exp_name"],
        "cot": round(cot, 2)
    })


# --- plot cost of transport ---------------------------

# exp_names = [r["name"] for r in results]
# cot_values = [r["cot"] for r in results]

# fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# ax.plot(exp_names, cot_values, 'k-')
# ax.set_xlabel("Experiment")
# ax.set_ylabel("Cost of Transport")
# plt.show()

# --- plot cot as a diverging bar chart
# exp_names = [r["name"] for r in results if "baseline" not in r["name"]]
# cot_values = [r["cot"] for r in results if "baseline" not in r["name"]]
# baseline = [r["cot"] for r in results if "baseline" in r["name"]][0]

# deltas = [v - baseline for v in cot_values]

# fig, ax = plt.subplots(figsize=(13, 5))

# ax.barh(exp_names, deltas, left=baseline)
# ax.axvline(baseline, color='red', linewidth=3.5, linestyle='-')

# ax.set_ylabel('Experiment', fontsize=12)
# ax.set_xlabel('Cost of Transport', fontsize=12)

# plt.show()