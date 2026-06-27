import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cost_of_transport import parse_distribution, find_distance_covered

BASELINE_FILE = "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv"

# FILES = [
#     # All NPA
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline"},

#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA"},

#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA"},
#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA"},

#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_NPA/incline_flat_stack_center_8kg_NPA_joints_20260428_132539.csv", "exp_name": "Stack Center NPA"},

#     # All PA
#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA"},
#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_PA/incline_flat_stack_center_8kg_PA_joints_20260428_131820.csv", "exp_name": "Stack Center PA"},
# ]

# INCLINE FLAT TERRAIN  EXPERIMENTS

FILES = [
    # All NPA
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline_incline_flat", "mass": 33.8, "loop": 3},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate NPA", "mass": 33.8+11.6, "distribution": "Front", "control_mode": "NPA", "position": "front", "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 33.8+11.6, "distribution": "Center", "control_mode": "NPA", "position": "center", "loop": 3},

    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA"},
    
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA", "mass": 33.8+11.6, "distribution": "Stack Front", "control_mode": "NPA", "position": "front", "loop": 1},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_NPA/incline_flat_stack_center_8kg_NPA_joints_20260428_132539.csv", "exp_name": "Stack Center NPA", "mass": 33.8+11.6, "distribution": "Stack Center", "control_mode": "NPA", "position": "center", "loop": 3},

    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_NPA/incline_flat_8kg_adj_center_NPA_joints_20260426_145657.csv", "exp_name": "Adjacent Center NPA", "mass": 33.8+13.6, "distribution": "Adjacent Center", "control_mode": "NPA", "position": "center", "loop": 3},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_crossover_May_11/water_center_crate_8kg_NPA/incline_crossover_center_crate_water_8kg_NPA_joints_20260511_164738.csv", "exp_name": "Center Crate NPA Water", "mass": 33.8+11.6, "distribution": "Center", "control_mode": "NPA", "position": "center", "loop": 3},

    # All PA
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA", "mass": 33.8+11.6, "distribution": "Front", "control_mode": "PA", "position": "front", "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8+11.6, "distribution": "Center", "control_mode": "PA", "position": "center", "loop": 3},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA", "mass": 33.8+11.6, "distribution": "Stack Front", "control_mode": "PA" , "position": "front", "loop": 3},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_8kg_PA/incline_flat_stack_center_8kg_PA_joints_20260428_131820.csv", "exp_name": "Stack Center PA", "mass": 33.8+11.6, "distribution": "Stack Center", "control_mode": "PA", "position": "center", "loop": 3},

    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent Center PA", "mass": 33.8+13.6, "distribution": "Adjacent Center", "control_mode": "PA", "position": "center", "loop": 3},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_crossover_May_11/water_center_crate_8kg_PA/incline_crossover_center_crate_water_8kg_PA_joints_20260511_165320.csv", "exp_name": "Center Crate PA Water", "mass": 33.8+11.6, "distribution": "Center", "control_mode": "PA", "position": "center", "loop": 3},

]

LEGS   = ["fl", "fr", "hl", "hr"]
# JOINTS = ["hx", "hy", "kn"]
JOINTS = ["hx"]

JOINT_POS_COLS = [f"{leg}.{joint}_position" for leg in LEGS for joint in JOINTS]


def find_limits():
    df = pd.read_csv(BASELINE_FILE)
    limits = []
    for col in JOINT_POS_COLS:
        positions = df[col].to_numpy()
        # print("min: ", np.min(positions), " max: ", np.max(positions))
        limits.append({
            "joint_col_name": col,
            "lower_limit": np.percentile(positions, 5),
            "upper_limit": np.percentile(positions, 95),
        })
    # print(limits)
    return limits

# Note: Use this function when I need to only consider moving points
def find_duration(df):
    x_velocity = df["vel_of_body_in_vision_lin_x"]
    y_velocity = df["vel_of_body_in_vision_lin_y"]

    # velocity where robot is not static
    vel = np.sqrt(x_velocity*x_velocity + y_velocity*y_velocity)
    vel_moving = df[vel > 0.2]
    return len(vel_moving)

FS = 333

def compute_vibration_cost(df, limits):
    n_samples = len(df)
    duration_s = n_samples / FS
    # duration_s = find_duration(df) / FS
    distance_covered = find_distance_covered(df)
    total_cost = 0.0
    for entry in limits:
        col   = entry["joint_col_name"]
        lower = entry["lower_limit"]
        upper = entry["upper_limit"]

        positions = df[col].to_numpy()
        below = np.where(positions < lower, np.abs(positions - lower), 0.0)
        above = np.where(positions > upper, np.abs(positions - upper), 0.0)
        total_cost += (below + above).sum()

    return total_cost , total_cost / duration_s, total_cost / distance_covered


def distribution_vs_vibration(results):
    series = {
        "Front_NPA": {},
        "Front_PA": {},
        "Center_NPA": {},
        "Center_PA": {}
    }

    dist_order = ["Standard", "Stack", "Adjacent"]
    x_pos = {d: i for i, d in enumerate(dist_order)} # "Standard": 0, "Stack": 1, "Adjacent": 2
 
    for result in results:
        dist_type, position = parse_distribution(result["distribution"])

        key = f"{position}_{result['control_mode']}"
        if key in series:
            series[key][dist_type] = result["vibration_cost"] / result["loop"]

    def _xy(key):
        pts = series[key]
        keys = sorted(pts, key=lambda d: x_pos[d])
        return [x_pos[d] for d in keys], [pts[d] for d in keys] # returns x and y array for plot

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(*_xy("Front_NPA"),  "-o",  color="black",   label="Front_NPA")
    ax.plot(*_xy("Front_PA"),   "--o", color="black",   label="Front_PA")
    ax.plot(*_xy("Center_NPA"), "-s",  color="red", label="Center_NPA")
    ax.plot(*_xy("Center_PA"),  "--s", color="red", label="Center_PA")

    ax.set_xticks(range(len(dist_order)))
    ax.set_xticklabels(dist_order)
    ax.set_xlabel("Distribution")
    ax.set_ylabel("Vibration Cost (per trial)")
    ax.legend()
    ax.grid(True)

    fig.suptitle("")

    ax.axhline(y=2031, color="grey", linestyle='-', linewidth=1.0)
    ax.annotate('Baseline', 
            xy=(1.4, 2031), 
            xytext=(1.5, 3000),
            arrowprops=dict(facecolor='black', shrink=0.005),
            fontsize=10,
            color='black')
    
    plt.tight_layout()
    plt.savefig("vibration_cost.png", dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    limits = find_limits()
    results = []

    for file in FILES:
        df = pd.read_csv(file["data_file"])

        if "start" in file:
            df = df.iloc[file["start"]: file["end"]]

        cost, cost_per_second, cost_per_unit_distance = compute_vibration_cost(df, limits)
        print(f"[{file['exp_name']}] Vibration Cost : {cost:.4f}")
        if "distribution" in file and "control_mode" in file:
            results.append({
                "name": file["exp_name"],
                "vibration_cost": round(cost, 4),
                "vibration_cost_per_second": round(cost_per_second, 4),
                "vibration_cost_per_unit_distance": round(cost_per_unit_distance, 4),
                "distribution": file["distribution"],
                "control_mode": file["control_mode"],
                "loop": file["loop"]
            })

    
    distribution_vs_vibration(results)

