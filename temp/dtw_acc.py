from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dtw import dtw

# IMU_FILES = [
#     # {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_imu_20260331_111335.csv", "exp_name": "baseline", "mass": 33.8},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_imu_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_NPA/incline_flat_8kg_stack_center_crate_NPA_imu_20260413_155829.csv", "exp_name": "Stack Center NPA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_imu_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_PA/incline_flat_8kg_stack_center_crate_PA_imu_20260413_160240.csv", "exp_name": "Stack Center PA", "mass": 33.8 + 11.6},
# ]

# FILES = [
#     # {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_imu_20260331_111335.csv", "exp_name": "baseline", "mass": 33.8},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_NPA/incline_flat_8kg_stack_center_crate_NPA_joints_20260413_155829.csv", "exp_name": "Stack Center NPA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_joints_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_PA/incline_flat_8kg_stack_center_crate_PA_joints_20260413_160240.csv", "exp_name": "Stack Center PA", "mass": 33.8 + 11.6},
# ]

file1 = "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_joints_20260413_142042.csv"
file2 = "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_NPA/incline_flat_8kg_stack_center_crate_NPA_joints_20260413_155829.csv"

COL = "vel_of_body_in_vision_lin_x"
DOWN_SAMPLE = 20

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

velocity_x_1 = df1[COL].dropna().to_numpy()[::DOWN_SAMPLE]
velocity_x_2 = df2[COL].dropna().to_numpy()[::DOWN_SAMPLE]

print(f"Signal lengths after downsampling: {len(velocity_x_1)}, {len(velocity_x_2)}")

alignment = dtw(velocity_x_1, velocity_x_2, keep_internals=True)

# 3. Visualize the alignment (connection) between signals
# This creates a plot showing how points in 'x' map to points in 'y'
alignment.plot(type="twoway", offset=2)
plt.show()
