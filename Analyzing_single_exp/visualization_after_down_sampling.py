import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

DATA_FILE = "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv"
DATA_FILE_IMU = ""
df = pd.read_csv(DATA_FILE)
df_size = df["vel_of_body_in_vision_lin_x"].size
print(df_size,"before")

down_sampled_df = df.iloc[::10]

down_sampled_df_size = down_sampled_df["vel_of_body_in_vision_lin_x"].size
print(down_sampled_df_size, "after")

# calculate the sudden acc jump
velocity_column = down_sampled_df["vel_of_body_in_vision_lin_x"]
acceleration_from_velocity = np.diff(velocity_column) / 0.3
threshold = 0.8

jump_indices = np.where(acceleration_from_velocity > threshold)[0]
# --------

experiment_time_data = np.arange(0, down_sampled_df_size)
x_position = down_sampled_df["vision_tform_body_pos_x"]
y_position = down_sampled_df["vision_tform_body_pos_y"]

x_velocity = down_sampled_df["vel_of_body_in_vision_lin_x"]
y_velocity = down_sampled_df["vel_of_body_in_vision_lin_y"]

# plt.axis('equal')
# Figure 1: to visualize x and y position independetly 
fig, axs = plt.subplots(2, 1, figsize=(18, 10))

axs[0].plot(experiment_time_data, x_position, '-')
axs[0].set_title('x_position and x_velocity')
axs[0].set_ylabel("position")

axs[1].plot(experiment_time_data, y_position, '-')
axs[1].set_title('y_position and y_velocity')
axs[1].set_ylabel("position")

for idx in jump_indices:
    axs[0].axvline(x=idx, color='black', linestyle='--')


# 5. Plot the second data series on the right y-axis
ax2 = axs[0].twinx() 
color = 'tab:red'
ax2.set_ylabel('velocity', color=color)
ax2.plot(experiment_time_data, x_velocity, color=color)

ax3 = axs[1].twinx() 
color = 'tab:red'
ax3.set_ylabel('velocity', color=color)
ax3.plot(experiment_time_data, y_velocity, color=color)

# --- trim sliders ------------------------------------------------------------
fig.subplots_adjust(bottom=0.18)

ax_start = fig.add_axes([0.15, 0.08, 0.7, 0.03])
ax_end   = fig.add_axes([0.15, 0.03, 0.7, 0.03])

N = down_sampled_df_size - 1
slider_start = Slider(ax_start, 'Trim start', 0, N, valinit=0, valstep=1)
slider_end   = Slider(ax_end,   'Trim end',   0, N, valinit=N, valstep=1)

vline_start = [axs[0].axvline(x=0, color='green', linestyle='-', linewidth=1.5),
               axs[1].axvline(x=0, color='green', linestyle='-', linewidth=1.5)]
vline_end   = [axs[0].axvline(x=N, color='blue',  linestyle='-', linewidth=1.5),
               axs[1].axvline(x=N, color='blue',  linestyle='-', linewidth=1.5)]

def update(_):
    s = int(slider_start.val)
    e = int(slider_end.val)
    for line in vline_start:
        line.set_xdata([s, s])
    for line in vline_end:
        line.set_xdata([e, e])
    fig.canvas.draw_idle()

slider_start.on_changed(update)
slider_end.on_changed(update)

def on_close(_):
    print(f"\n>>> Trim values (downsampled counts):")
    print(f"    start = {int(slider_start.val)}")
    print(f"    end   = {int(slider_end.val)}")
    print(f">>> Raw row equivalents (x{10} downsample):")
    print(f"    start = {int(slider_start.val) * 10}")
    print(f"    end   = {int(slider_end.val) * 10}")

fig.canvas.mpl_connect('close_event', on_close)

plt.show()

