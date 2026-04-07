import numpy as np
import pandas as pd

DATA_FILE = "/home/nerve/Desktop/data_collected/temp/loop_test_without_time_joints_20260316_120802.csv"

df = pd.read_csv(DATA_FILE)

# Downsample the data



# After Down sampling

velocity_column = df["vel_of_body_in_vision_lin_x"]

acceleration_from_velocity = np.diff(velocity_column) / 0.003

# print(acceleration_from_velocity[3000])

threshold = 4

jump_indices = np.where(acceleration_from_velocity > threshold)

print(len(jump_indices[0]))
print(jump_indices)