import matplotlib.pyplot as plt

# Data
labels = ['Group A', 'Group B', 'Group C', 'Group D']
values = [85, 40, 60, 30]
baseline = 50

# Calculate the movement from baseline
# (Positive = right, Negative = left)
deltas = [v - baseline for v in values]

fig, ax = plt.subplots()

# Create the bars
# We use 'baseline' as the starting point for every bar
ax.barh(labels, deltas, left=baseline, color=['green' if d > 0 else 'red' for d in deltas])

# Add the vertical baseline for clarity
ax.axvline(baseline, color='black', linewidth=1.5, linestyle='--')

# Formatting
ax.set_xlabel('Value')
ax.set_title('Diverging Bar Chart (Baseline = 50)')

plt.show()