from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
DOWN_SAMPLE = 10
import numpy as np
DATA_FILE_IMU = "/home/nerve/Desktop/data_collected/temp/loop_test_without_time_imu_20260316_120802.csv"

# 1. Define Filter Requirements
# fs = int(1000 / DOWN_SAMPLE)  # Sampling frequency (1000Hz)
fs = int(1000)
cutoff = 10.0     # The frequency you decided on from the FFT
order = 2         # Order 1 attenuates very slowly

# 2. Create the filter coefficients
# Nyquist frequency is always half the sampling rate
nyq = 0.5 * fs
low = cutoff / nyq
b, a = butter(order, low, btype='low')

# getting raw IMU data

df = pd.read_csv(DATA_FILE_IMU)
df = df[:30000] # only first loop

## Down sampling then  filtering
# down_sampled_df = df.iloc[::DOWN_SAMPLE]
# down_sampled_df_size = down_sampled_df["acc_x"].size
# experiment_time_data = np.arange(0, down_sampled_df_size)
# print(down_sampled_df_size, "after")

# acc_x_raw = down_sampled_df["acc_x"]

# # -- Apply the filter
# filtered_accel = filtfilt(b, a, acc_x_raw)

# Filtering and then downsampling
acc_x_raw = df["acc_x"]
filtered_accel = filtfilt(b, a, acc_x_raw)
filtered_accel = filtered_accel[::DOWN_SAMPLE]
experiment_time_data = np.arange(0, len(filtered_accel))

# Drawing Figure
fig, ax = plt.subplots(1, 1, figsize=(18, 4))

ax.plot(experiment_time_data, filtered_accel, '-')
ax.set_xlabel("time step")
ax.set_ylabel("filterd acc_x")

# ax2 = ax.twinx() 
# color = 'tab:red'
# ax2.set_ylabel('raw acc_x', color=color)
# ax2.plot(experiment_time_data, acc_x_raw, color=color, linestyle='--')

plt.show()