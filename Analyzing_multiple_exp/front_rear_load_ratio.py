import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# FILES_NPA_8KG = [
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate", "control_mode": "NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate", "control_mode": "NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA", "control_mode": "NPA"},
# ]

    # {"data_file": "", "exp_name": "", "control_mode": "NPA"},

BASELINE =  {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "Baseline", "control_mode": "NPA"},
# 7.6 kg
FILES_low_weight_sand = [
    # {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "Baseline", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb4/flat_terrain_5.2kg_front_crate/exp1/flat_terrain_5.2kg_front_crate_exp1.csv", "exp_name": "Front Crate", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb4/flat_terrain_5.2kg_center_crate/exp1/flat_terrain_5.2kg_center_crate_exp1.csv", "exp_name": "Center Crate", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb4/flat_terrain_5.2kg_rear_crate/exp1/flat_terrain_5.2kg_rear_crate_exp1.csv", "exp_name": "Rear Crate", "control_mode": "NPA"},
]

# FILES_high_weight_water = [
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Apr_7/water_center/water_center_8kg_joints_20260407_195525.csv", "exp_name": "Center Crate", "control_mode": "NPA"},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Apr_7/water_front/water_front_8kg_joints_20260407_200509.csv", "exp_name": "Front Crate", "control_mode": "NPA"},
# ]

FILES_high_weight_sand_flat = [
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_frontcrate_m_8kg/exp2/flat_frontcrate_m_8kg_exp2_joints.csv", "exp_name": "Front Crate", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_joints.csv", "exp_name": "Center Crate", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_rearcrate_m_8kg/exp1/flat_rearcrate_m_8kg_exp1_joints.csv", "exp_name": "Rear Crate", "control_mode": "NPA"},
]

# incline flat terrain
FILES_high_weight_sand_incline_flat = [
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front Crate", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate", "control_mode": "NPA"},
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_June_11/incline_flat_optimal_8kg/incline_flat_optimal_distribution_T_11.6kg_NPA_joints_20260611_152231.csv", "exp_name": "Optimal", "control_mode": "NPA"},
]

front_joints = ["fl.hx_load", "fl.hy_load", "fl.kn_load", "fr.hx_load", "fr.hy_load", "fr.kn_load"]
rear_joints = ["hl.hx_load", "hl.hy_load", "hl.kn_load", "hr.hx_load", "hr.hy_load", "hr.kn_load"]

def find_front_rear_load_ratio(df: pd.DataFrame, front_joints: list[str], rear_joints: list[str]):
    df_front_joint_loads_matrix = df[front_joints].to_numpy()
    df_rear_joint_loads_matrix = df[rear_joints].to_numpy()

    front_load = np.sum(np.abs(df_front_joint_loads_matrix))
    rear_load = np.sum(np.abs(df_rear_joint_loads_matrix))
    fr_load_ratio = ( front_load - rear_load ) / ( front_load + rear_load )
    # fr_load_ratio = ( front_load  ) / ( rear_load )

    return fr_load_ratio


def plot_front_rear_load_ratio(
    load_ratio_low_weight_sand: list[dict],
    load_ratio_high_weight_sand: list[dict],
    load_ratio_high_weight_water: list[dict],
    save_path: str = None,
):
    x_labels = ["Front Crate", "Center Crate", "Rear Crate"]
    x_pos = np.arange(len(x_labels))

    groups = [
        ("Sand (7.6 kg)", load_ratio_low_weight_sand, "tab:blue"),
        ("Sand (11.6 kg)",  load_ratio_high_weight_sand, "tab:orange"),
        # ("High Weight Water (11.6 kg)", load_ratio_high_weight_water, "tab:green"),
    ]

    _, ax = plt.subplots(figsize=(6, 3))

    for label, results, color in groups:
        ratio_map = {r["exp_name"]: r["fr_load_ratio"] for r in results}
        y_vals = [ratio_map.get(name, np.nan) for name in x_labels]
        ax.plot(x_pos, y_vals, marker="o", label=label, color=color)
        # for xi, yi in zip(x_pos, y_vals):
        #     if not np.isnan(yi):
        #         ax.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(0, 6),
        #                     textcoords="offset points", ha="center", fontsize=8, color=color)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="gray", linewidth=1.2, linestyle="-")
    # ax.set_xlabel("Crate Position")
    ax.set_ylabel("Front-Rear Load Ratio", fontsize= 12)
    # ax.set_title("Front-Rear Load Ratio by Crate Position and Payload")
    ax.legend()
    ax.grid(axis="y", linestyle="-", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.show()

    else:
        plt.show()


def plot_front_rear_load_ratio_terrain(
    load_ratio_high_weight_sand_flat: list[dict],
    load_ratio_high_weight_sand_incline_flat: list[dict],
    save_path: str = None,
):  
    # x_labels = ["Front Crate", "Center Crate", "Rear Crate"]
    x_labels = ["Front", "Optimal", "Center", "Rear"]
    distributions = ["Front Crate", "Optimal", "Center Crate", "Rear Crate"]

    x_pos = [-33.0, -4.86, 0.0, 33.0]
    # x_pos = np.arange(len(x_labels))

    groups = [
        # ("Flat Terrain (11.6 kg)", load_ratio_high_weight_sand_flat, "tab:blue"),
        ("Incline Flat Terrain (11.6 kg)",  load_ratio_high_weight_sand_incline_flat, "tab:orange"),
    ]

    _, ax = plt.subplots(figsize=(8, 4.5))

    for label, results, color in groups:
        ratio_map = {r["exp_name"]: r["fr_load_ratio"] for r in results}
        y_vals = [ratio_map.get(name, np.nan) for name in distributions]
        ax.plot(x_pos, y_vals, marker="o", label=label, color=color)
        # for xi, yi in zip(x_pos, y_vals):
        #     if not np.isnan(yi):
        #         ax.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(0, 6),
        #                     textcoords="offset points", ha="center", fontsize=8, color=color)

    ax.set_xticks(x_pos)
    ax.tick_params(axis="x", width=2, length=6)
    ax.set_xticklabels(x_labels, rotation=30, fontsize=12)
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="gray", linewidth=1.2, linestyle="-")
    # ax.set_xlabel("Crate Position")
    ax.set_ylabel("Front-Rear Load Ratio", fontsize=12)
    # ax.set_title("Front-Rear Load Ratio by Crate Position and Payload")
    ax.legend()
    ax.grid(linestyle="-", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.show()

    else:
        plt.show()


if __name__ == "__main__":

    load_ratio_low_weight_sand = []
    # load_ratio_high_weight_sand = []
    load_ratio_high_weight_water = []
    
    load_ratio_high_weight_sand_incline_flat = []
    load_ratio_high_weight_sand_flat = []

    for file in FILES_low_weight_sand:
        df = pd.read_csv(file["data_file"])
        fr_load_ratio = find_front_rear_load_ratio(df=df, front_joints=front_joints, rear_joints=rear_joints)

        load_ratio_low_weight_sand.append({
            "exp_name": file["exp_name"],
            "fr_load_ratio": fr_load_ratio
        })

    for file in FILES_high_weight_sand_flat:
        df = pd.read_csv(file["data_file"])
        fr_load_ratio = find_front_rear_load_ratio(df=df, front_joints=front_joints, rear_joints=rear_joints)

        load_ratio_high_weight_sand_flat.append({
            "exp_name": file["exp_name"],
            "fr_load_ratio": fr_load_ratio
        })
    
    for file in FILES_high_weight_sand_incline_flat:
        df = pd.read_csv(file["data_file"])
        fr_load_ratio = find_front_rear_load_ratio(df=df, front_joints=front_joints, rear_joints=rear_joints)

        load_ratio_high_weight_sand_incline_flat.append({
            "exp_name": file["exp_name"],
            "fr_load_ratio": fr_load_ratio
        })

    # for file in FILES_high_weight_water:
    #     df = pd.read_csv(file["data_file"])
    #     fr_load_ratio = find_front_rear_load_ratio(df=df, front_joints=front_joints, rear_joints=rear_joints)

    #     load_ratio_high_weight_water.append({
    #         "exp_name": file["exp_name"],
    #         "fr_load_ratio": fr_load_ratio
    #     })
    

    # plot_front_rear_load_ratio(
    #     load_ratio_low_weight_sand=load_ratio_low_weight_sand,
    #     load_ratio_high_weight_sand=load_ratio_high_weight_sand_flat,
    #     load_ratio_high_weight_water=load_ratio_high_weight_water,
    #     save_path="front_rear_load_ratio.png",
    # )

    plot_front_rear_load_ratio_terrain(
        load_ratio_high_weight_sand_flat=load_ratio_high_weight_sand_flat,
        load_ratio_high_weight_sand_incline_flat=load_ratio_high_weight_sand_incline_flat,
        save_path="front_rear_load_ratio_terrain.png",
    )

