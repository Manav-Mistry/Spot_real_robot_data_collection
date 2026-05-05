import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FILES = [
    # All NPA
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline", "mass": 33.8}, 
   
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_joints_20260413_140451.csv", "exp_name": "Front  Crate NPA", "mass": 33.8+ 11.6},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 33.8+ 11.6},

    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_joints_20260413_142659.csv", "exp_name": "Rear Crate NPA", "mass": 33.8+ 11.6},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_joints_20260413_152852.csv", "exp_name": "Stack Front NPA", "mass": 33.8+ 11.6}, 

    {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_NPA/incline_flat_8kg_stack_center_crate_NPA_joints_20260413_155829.csv", "exp_name": "Stack Center NPA", "mass": 33.8+ 11.6},

    # # All PA
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_joints_20260413_150510.csv", "exp_name": "Front Crate PA", "mass": 33.8 + 11.6},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8 + 11.6},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_joints_20260413_153830.csv", "exp_name": "Stack Front PA", "mass": 33.8 + 11.6},
    # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_PA/incline_flat_8kg_stack_center_crate_PA_joints_20260413_160240.csv", "exp_name": "Stack Center PA", "mass": 33.8 + 11.6},
    # {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_27/incline_flat_center_crate_16kg_PA_joints_20260427_151500.csv", "exp_name": "20kg Center PA", "mass": 33.8 + 11.6},

   

]

total_plots = len(FILES)

results = []

for file in FILES:
    df = pd.read_csv(file["data_file"])
    
    if "start" in file:
        start = file["start"]
        end = file["end"]
        df = df.iloc[start: end]
  
    x_position = df["vision_tform_body_pos_x"]
    y_position = df["vision_tform_body_pos_y"]
    z_position = df["vision_tform_body_pos_z"]

    results.append({
        "name": file["exp_name"],
        "x": x_position,
        "y": y_position,
        "z": z_position
    })


for result in results:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')

    x = result["x"].to_numpy()
    y = result["y"].to_numpy()
    z = result["z"].to_numpy()


    ax.plot(x, y, z)

    ax.scatter(x[0], y[0], z[0], color="green", s=80, zorder=5)
    ax.text(x[0], y[0], z[0], "Start", fontsize=9, color="green")

    ax.scatter(x[-1], y[-1], z[-1], color="red", s=80, zorder=5)
    ax.text(x[-1], y[-1], z[-1], "End", fontsize=9, color="red")

    ax.set_title(result["name"])
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Z position (m)")
    
    ax.set_aspect("equal")
    ax.grid(True)

    plt.tight_layout()

plt.show()
