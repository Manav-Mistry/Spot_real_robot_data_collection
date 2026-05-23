import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


FILES = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8},
]

for file in FILES:
    df = pd.read_csv(file["data_file"])

    if "start" in file:
        df = df.iloc[file["start"]: file["end"]]

    x = df["vision_tform_body_pos_x"].to_numpy()
    y = df["vision_tform_body_pos_y"].to_numpy()
    z = df["vision_tform_body_pos_z"].to_numpy()

    step_dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    cum_dist = np.concatenate([[0], np.cumsum(step_dist)])

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)

    ax.plot(cum_dist, z, color="steelblue", linewidth=1.5)

    vline = ax.axvline(x=cum_dist[0], color="red", linewidth=1.2, linestyle="--")
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                    verticalalignment="top", color="red")

    ax.set_title(f"{file['exp_name']} — Elevation Profile")
    ax.set_xlabel("Cumulative distance traveled (m)")
    ax.set_ylabel("Z position (m)")
    ax.grid(True)

    ax_slider = plt.axes([0.12, 0.05, 0.78, 0.04])
    slider = Slider(ax_slider, "Distance (m)", cum_dist[0], cum_dist[-1],
                    valinit=cum_dist[0], valstep=(cum_dist[-1] - cum_dist[0]) / 1000)

    def update(val):
        d = val
        idx = np.searchsorted(cum_dist, d)
        idx = min(idx, len(z) - 1)
        vline.set_xdata([d, d])
        label.set_text(f"dist = {d:.2f} m   z = {z[idx]:.3f} m")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(cum_dist[0])

plt.show()
