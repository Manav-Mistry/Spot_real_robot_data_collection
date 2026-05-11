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

# INCLINE FLAT TERRAIN  EXPERIMENTS

FILES = [
    # All NPA
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate NPA", "mass": 33.8+11.6, "distribution": "Front", "control_mode": "NPA", "position": "front"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 33.8+11.6, "distribution": "Center", "control_mode": "NPA", "position": "center"},

    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA"},
    
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA", "mass": 33.8+11.6, "distribution": "Stack Front", "control_mode": "NPA", "position": "front"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_NPA/incline_flat_stack_center_8kg_NPA_joints_20260428_132539.csv", "exp_name": "Stack Center NPA", "mass": 33.8+11.6, "distribution": "Stack Center", "control_mode": "NPA", "position": "center"},

    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent Center NPA", "mass": 33.8+13.6, "distribution": "Adjacent Center", "control_mode": "NPA", "position": "center"},

    # All PA
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA", "mass": 33.8+11.6, "distribution": "Front", "control_mode": "PA", "position": "front"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8+11.6, "distribution": "Center", "control_mode": "PA", "position": "center"},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA", "mass": 33.8+11.6, "distribution": "Stack Front", "control_mode": "PA" , "position": "front"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_PA/incline_flat_stack_center_8kg_PA_joints_20260428_131820.csv", "exp_name": "Stack Center PA", "mass": 33.8+11.6, "distribution": "Stack Center", "control_mode": "PA", "position": "center"},

    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent Center PA", "mass": 33.8+13.6, "distribution": "Adjacent Center", "control_mode": "PA", "position": "center"},
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
    v_threshold = 0.25
    vel = vel[vel > v_threshold]

    return np.mean(vel)


def cot_diverging_bar_chart(results):
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

    # plt.savefig("cost_of_transport.png", dpi=300, bbox_inches='tight')
    plt.show()


def parse_distribution(dist_str):
    """Split a distribution string like 'Stack Front' into (dist_type, position)."""
    position = "Front" if "Front" in dist_str else "Center"
    if "Stack" in dist_str:
        dist_type = "Stack"
    elif "Adjacent" in dist_str:
        dist_type = "Adjacent"
    else:
        dist_type = "Standard"
    return dist_type, position


def cot_comparison_plot(results_detailed):
    dist_order = ["Standard", "Stack", "Adjacent"]
    x_pos = {d: i for i, d in enumerate(dist_order)}

    series = {
        "Front_NPA": {},
        "Front_PA":  {},
        "Center_NPA": {},
        "Center_PA":  {},
    }

    for r in results_detailed:
        dist_type, position = parse_distribution(r["distribution"])
        key = f"{position}_{r['control_mode']}"
        if key in series:
            series[key][dist_type] = r["cot"]

    def _xy(key):
        pts = series[key]
        keys = sorted(pts, key=lambda d: x_pos[d])
        return [x_pos[d] for d in keys], [pts[d] for d in keys]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(*_xy("Front_NPA"),  "-o",  color="black",   label="Front_NPA")
    ax.plot(*_xy("Front_PA"),   "--o", color="black",   label="Front_PA")
    ax.plot(*_xy("Center_NPA"), "-s",  color="red", label="Center_NPA")
    ax.plot(*_xy("Center_PA"),  "--s", color="red", label="Center_PA")

    ax.set_xticks(range(len(dist_order)))
    ax.set_xticklabels(dist_order)
    ax.set_xlabel("Distribution")
    ax.set_ylabel("Cost of Transport")
    ax.legend()
    ax.grid(True)

    # fig.suptitle("CoT by Distribution and Payload Position\nIncline Flat")
    ax.axhline(y=0.4233, color="grey", linestyle='-', linewidth=1.75)
    ax.annotate('Baseline', 
            xy=(1.4, 0.4233), 
            xytext=(1.5, 0.40),
            arrowprops=dict(facecolor='black', shrink=0.005),
            fontsize=10,
            color='black')
    plt.tight_layout()
    plt.savefig("cost_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    results = []
    results_detailed = []

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
            "cot": round(cot_energy, 3)
        })

        if "distribution" in file and "control_mode" in file:
            results_detailed.append({
                "name": file["exp_name"],
                "cot": round(cot_energy, 3),
                "distribution": file["distribution"],
                "control_mode": file["control_mode"],
            })

    # cot_diverging_bar_chart(results)
    cot_comparison_plot(results_detailed)

    # --- plot cost of transport ---------------------------

    # exp_names = [r["name"] for r in results]
    # cot_values = [r["cot"] for r in results]

    # fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # ax.plot(exp_names, cot_values, 'k-')
    # ax.set_xlabel("Experiment")
    # ax.set_ylabel("Cost of Transport")
    # plt.show()

    # --- plot cot as a diverging bar chart
   