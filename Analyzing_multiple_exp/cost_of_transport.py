import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ALL FLAT TERRAIN NPA EXPERIMENTS 
# FILES = [
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline", "mass": 33.8}, 
   
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/front/adj_single_tier_front_8kg_NPA_loop3_joints_20260320_125855.csv", "exp_name": "Adjacent Front", "mass": 33.8+ 13.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/center/adj_single_tier_center_8kg_NPA_loop3_joints_20260320_132519.csv", "exp_name": "Adjacent Center", "mass": 33.8+ 13.6},

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_center/double_tier_center_8kg_NPA_loop3_joints_20260320_162856.csv", "exp_name": "Stack Center", "mass": 33.8+ 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_front/double_tier_front_8kg_NPA_loop3_joints_20260320_151917.csv", "exp_name": "Stack Front", "mass": 33.8+ 11.6}, 

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/front_rear/single_tier_front_rear_8kg_NPA_loop3_joints_20260320_143151.csv", "exp_name": "Front & Rear (distributed)", "mass": 33.8+ 12.8},

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_joints.csv", "exp_name": "Center Crate", "mass": 33.8+ 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_frontcrate_m_8kg/exp1/flat_frontcrate_m_8kg_exp1_joints.csv", "exp_name": "Front Crate", "mass": 33.8+ 11.6},

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Apr_7/water_center/water_center_8kg_joints_20260407_195525.csv", "exp_name": "Water, Center Crate", "mass": 33.8+ 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Apr_7/water_front/water_front_8kg_joints_20260407_200509.csv", "exp_name": "Water, Front Crate", "mass": 33.8+ 11.6},

# ]

# INCLINE FLAT TERRAIN  EXPERIMENTS, baeline is from flat terrain

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
FS = 333
DT = 1/FS

TORQUE_COLS = [f"{leg}.{joint}_load" for joint in JOINTS for leg in LEGS]
ANG_VEL_COLS = [f"{leg}.{joint}_velocity" for joint in JOINTS for leg in LEGS]

TORQUE_VELOCITY_PAIR = list(zip(TORQUE_COLS, ANG_VEL_COLS))

def compute_power(df):
    power_joint_t = [ np.abs(df[t] * df[v]) for t, v in TORQUE_VELOCITY_PAIR ]
    # instantaneous_power = np.sum(power_joint_t, axis=0)  
    # moving_power = instantaneous_power[:-1][moving]      
    total_energy = np.sum(power_joint_t) * DT             
    return total_energy

def find_distance_covered(df):
    pos_x = df["vision_tform_body_pos_x"].to_numpy()
    pos_y = df["vision_tform_body_pos_y"].to_numpy()

    dx = np.diff(pos_x)
    dy = np.diff(pos_y)

    step_distances = np.sqrt(dx*dx + dy*dy)

    total_distance = np.sum(step_distances)
    total_time = len(step_distances) / FS

    return total_distance

def find_ave_velocity(df):
    vel_x = df["vel_of_body_in_vision_lin_x"]
    vel_y = df["vel_of_body_in_vision_lin_y"]

    vel = np.sqrt(vel_x*vel_x + vel_y*vel_y)
    v_threshold = 0.3
    vel = vel[vel > v_threshold]

    return np.mean(vel)


if __name__ == "__main__":
    results = []

    # --- find cost of transport ---------------------------
    for file in FILES:
        df = pd.read_csv(file["data_file"])
        
        if "start" in file:
            start = file["start"]
            end = file["end"]
            df = df.iloc[start: end]
        total_distance= find_distance_covered(df)
        total_energy = compute_power(df)

        cot_energy = total_energy / (file["mass"] * 9.8 * total_distance)

        print(f"[{file['exp_name']}] CoT (energy): {cot_energy:.4f} ")

        results.append({
            "name": file["exp_name"],
            "cot": round(cot_energy, 2)
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
    exp_names = [r["name"] for r in results if "baseline" not in r["name"]]
    cot_values = [r["cot"] for r in results if "baseline" not in r["name"]]
    baseline = [r["cot"] for r in results if "baseline" in r["name"]][0]

    deltas = [v - baseline for v in cot_values]

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.barh(exp_names, deltas, left=baseline)
    ax.axvline(baseline, color='red', linewidth=3.5, linestyle='-')

    ax.set_ylabel('Experiment', fontsize=12)
    ax.set_xlabel('Cost of Transport', fontsize=12)

    ax.set_title("CoT for Incline Flat [NPA vs PA]")

    plt.savefig("cost_of_transport.png", dpi=300, bbox_inches='tight')
    plt.show()