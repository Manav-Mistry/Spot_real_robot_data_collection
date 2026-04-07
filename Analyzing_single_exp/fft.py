import numpy as np
import matplotlib.pyplot as plt

# 1. Setup based on your IMU specs
fs = 1000  # Sampling frequency (1000Hz)
T = 1.0    # 1 second of data
t = np.linspace(0, T, fs, endpoint=False)

# 2. Simulate "Real" IMU data: 
# A 3Hz physical motion + some 60Hz electrical noise
accel_data = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

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