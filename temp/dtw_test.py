import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw

# 1. Generate two different signals
# Signal 1: Standard sine wave
x = np.sin(np.linspace(0, 6*np.pi, 100))

# Signal 2: Slightly shifted and stretched sine wave
y = np.sin(np.linspace(0.5, 6.5*np.pi, 120))

# 2. Compute the Dynamic Time Warping alignment
# Using Euclidean distance as the local cost measure
alignment = dtw(x, y, keep_internals=True)

# 3. Visualize the alignment (connection) between signals
# This creates a plot showing how points in 'x' map to points in 'y'
alignment.plot(type="twoway", offset=2)
plt.title("DTW Signal Alignment")
# plt.show()

# Print the final distance (similarity score)
print(f"DTW Distance: {alignment.distance}")

print(alignment.index1)
print(alignment.index2)