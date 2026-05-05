import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE_IMU = "/home/nerve/Desktop/data_collected/temp/loop_test_without_time_imu_20260316_120802.csv"
DOWN_SAMPLE = 10

df = pd.read_csv(DATA_FILE_IMU)

df_size = df["acc_x"].size
print(df_size,"before")

# down sampling
down_sampled_df = df.iloc[::DOWN_SAMPLE]
down_sampled_df_size = down_sampled_df["acc_x"].size
experiment_time_data = np.arange(0, down_sampled_df_size)
print(down_sampled_df_size, "after")

# 1. Setup based on your IMU specs
fs = int(1000 / DOWN_SAMPLE)  # Sampling frequency (1000Hz)
T = fs * down_sampled_df_size   # 1 second of data
t = np.linspace(0, T, fs, endpoint=False)

# IMU data: 
accel_data = down_sampled_df["gyro_z"]

# 3. Perform the FFT
# This converts the signal from the "Time Domain" to the "Frequency Domain"
fft_values = np.fft.fft(accel_data)
frequencies = np.fft.fftfreq(len(accel_data), 1/fs)

# We only care about the positive frequencies and the magnitude (strength)
mask = frequencies > 0
mag = np.abs(fft_values[mask])
freq_axis = frequencies[mask]

# 4. Find the "Dominant" frequency
peak_freq = freq_axis[np.argmax(mag)]
print(f"The primary frequency in the signal is: {peak_freq} Hz")

# 5. Plotting
plt.figure(figsize=(10, 4))
plt.plot(freq_axis, mag)
plt.title("Frequency Spectrum (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Strength/Magnitude")
plt.xlim(0, 100) # Zoom in to see the 3Hz and 60Hz peaks
plt.grid(True)
plt.show()