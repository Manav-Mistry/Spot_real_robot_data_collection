import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FILES = [
#     # {"filename": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "experiment_name": "stack_front_PA"},
#     {"filename": "/home/nerve/Desktop/data_collected/flat_Apr_7/water_center/water_center_8kg_joints_20260407_195525.csv", "experiment_name": "water_center_NPA"},
#     {"filename": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp2/flat_centercrate_m_8kg_exp2_joints.csv", "experiment_name": "center_crate_NPA"},


# ]

FILES = [
    # All NPA
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline", "mass": 33.8}, 
   
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front  Crate NPA", "mass": 33.8+ 11.6},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 33.8+ 11.6},

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA", "mass": 33.8+ 11.6},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA", "mass": 33.8+ 11.6}, 

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_NPA/incline_flat_8kg_stack_center_crate_NPA_joints_20260413_155829.csv", "exp_name": "Stack Center NPA", "mass": 33.8+ 11.6},

    # All PA
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA", "mass": 33.8 + 11.6},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8 + 11.6},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA", "mass": 33.8 + 11.6},
    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_PA/incline_flat_8kg_stack_center_crate_PA_joints_20260413_160240.csv", "exp_name": "Stack Center PA", "mass": 33.8 + 11.6},

]

results = []

for file in FILES:
    df = pd.read_csv(file["data_file"])
    z_trajectory = df["vision_tform_body_pos_z"].to_numpy()
    results.append({
        "exp_name": file["exp_name"],
        "z_trajectory": z_trajectory
    })


fig, axe = plt.subplots(1, 1, figsize=(10, 4))

for result in results:
    axe.plot(result["z_trajectory"], label=result["exp_name"])

axe.set_ylabel("Z axis (meter)")
axe.legend()
plt.grid(True)
plt.show()