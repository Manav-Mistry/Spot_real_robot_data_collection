import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE_IMU = "/home/nerve/Desktop/data_collected/temp/loop_test_without_time_imu_20260316_120802.csv"

df = pd.read_csv(DATA_FILE_IMU)

# Before after down sampling 

df_size = df["acc_x"].size
print(df_size,"before")

down_sampled_df = df.iloc[::10]
down_sampled_df_size = down_sampled_df["acc_x"].size
experiment_time_data = np.arange(0, down_sampled_df_size)
print(down_sampled_df_size, "after")

# acc_x, acc_y, acc_z

acc_x = down_sampled_df["acc_x"]
acc_y = down_sampled_df["acc_y"]
acc_z = down_sampled_df["acc_z"]

fig, axs = plt.subplots(3, 1, figsize=(18, 10))

axs[0].plot(experiment_time_data, acc_x, '-')
# axs[0].set_title('x_acceleration')
axs[0].set_xlabel("x_acceleration")

axs[1].plot(experiment_time_data, acc_y, '-')
# axs[1].set_title('y_position and y_velocity')
axs[1].set_xlabel("y_acceleration")

axs[2].plot(experiment_time_data, acc_z, '-')
# axs[2].set_title('y_position and y_velocity')
axs[2].set_xlabel("z_acceleration")

plt.show()
