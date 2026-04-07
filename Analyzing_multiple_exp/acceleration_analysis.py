from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


FILES = [
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/front/adj_single_tier_front_8kg_NPA_loop3_imu_20260320_125855.csv", "exp_name": "Adj_front_8kg", "start": 500*3, "end": 10500*3},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/center/adj_single_tier_center_8kg_NPA_loop3_imu_20260320_132519.csv", "exp_name": "Adj_center_8kg", "start": 2000*3, "end": 10700*3},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_center/double_tier_center_8kg_NPA_loop3_imu_20260320_162856.csv", "exp_name": "Stack_center_8kg", "start": 1500*3, "end": 10000*3},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_front/double_tier_front_8kg_NPA_loop3_imu_20260320_151917.csv", "exp_name": "Stack_front_8kg", "start": 2000*3, "end": 12000*3},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/front_rear/single_tier_front_rear_8kg_NPA_loop3_imu_20260320_143151.csv", "exp_name": "front_rear_8kg", "start": 1600*3, "end": 10600*3},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_imu.csv", "exp_name": "center_crate_8kg", "start": 2000*3, "end": 10500*3},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_frontcrate_m_8kg/exp1/flat_frontcrate_m_8kg_exp1_imu.csv", "exp_name": "front_crate_8kg", "start": 1500*3, "end": 11000*3},

    # {"data_file": "", "exp_name": "baseline", "start": , "end": },

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_imu_20260331_111335.csv", "exp_name": "baseline", "start": 11240*3, "end": 19700*3}, 


]

DOWN_SAMPLE = 10
CUTOFF_HZ = 10.0   # must be < Nyquist (fs/2); keep < fs/2 when changing DOWN_SAMPLE
FILTER_ORDER = 2
MAX_SAMPLES = None   # set to an integer (e.g. 5000) to limit, or None for all data
ACCEL_COLS = ["acc_x", "acc_y", "acc_z"]


# --- helper functions --------------------------------------------------------

def load_and_downsample(file_path, downsample, max_samples=MAX_SAMPLES, start=None, end=None):
    df = pd.read_csv(file_path)
    if start is not None or end is not None:
        df = df.iloc[start:end]
    df = df[::downsample]
    if max_samples is not None:
        df = df[:max_samples]
    return df.reset_index(drop=True)


def make_lowpass_filter(cutoff_hz, fs, order=FILTER_ORDER):
    nyq = 0.5 * fs
    assert 0 < cutoff_hz < nyq, (
        f"cutoff {cutoff_hz} Hz must be between 0 and Nyquist {nyq} Hz"
    )
    return butter(order, cutoff_hz / nyq, btype='low')


def apply_filter(b, a, df, columns=ACCEL_COLS):
    return {col: filtfilt(b, a, df[col].values) for col in columns}


def compute_jerk(accel_array, fs):
    """Jerk = time derivative of acceleration (m/s^3)."""
    return np.diff(accel_array) * fs


def mean_abs_jerk(jerk_array):
    return np.mean(np.abs(jerk_array))


# --- processing --------------------------------------------------------------

fs = int(1000 / DOWN_SAMPLE)
b, a = make_lowpass_filter(CUTOFF_HZ, fs)

results = []
for file in FILES:
    df = load_and_downsample(file["data_file"], DOWN_SAMPLE, start=file.get("start"), end=file.get("end"))
    filtered = apply_filter(b, a, df)
    jerk = {col: compute_jerk(filtered[col], fs) for col in ACCEL_COLS}
    maj = {col: mean_abs_jerk(jerk[col]) for col in ACCEL_COLS}

    results.append({
        "name": file["exp_name"],
        "filtered": filtered,
        "jerk": jerk,
        "maj": maj,
    })


# --- plot 1: filtered acceleration per experiment ----------------------------

# for result in results:
#     n_samples = len(result["filtered"][ACCEL_COLS[0]])
#     time = np.arange(n_samples) / fs

#     fig, axes = plt.subplots(3, 1, figsize=(10, 6))
#     fig.suptitle(f"Filtered Acceleration — {result['name']}")

#     for ax, col in zip(axes, ACCEL_COLS):
#         ax.plot(time, result["filtered"][col])
#         ax.set_ylabel(col)

#     fig.supxlabel("Time (s)")
#     plt.tight_layout()


# --- plot 2: mean absolute jerk comparison across experiments in a single 1x1 plot ----------------

# exp_names = [r["name"] for r in results]
# x = np.arange(len(exp_names))
# width = 0.25

# fig, ax = plt.subplots(figsize=(8, 5))
# for i, col in enumerate(ACCEL_COLS):
#     maj_values = [r["maj"][col] for r in results]
#     ax.bar(x + i * width, maj_values, width, label=col)

# ax.set_xticks(x + width)
# ax.set_xticklabels(exp_names, rotation=15, ha='right')
# ax.set_ylabel("Mean Absolute Jerk (m/s^3)")
# ax.set_title("Jerk Comparison Across Experiments")
# ax.legend()
# plt.tight_layout()

# plt.show()

# --- plot 3: diverging bar chart for mean absolute jerk with 3 subplots for x, y, and z

exp_names = [r["name"] for r in results if "baseline" not in r["name"]]

maj_x = [r["maj"][col] for r in results for col in r["maj"] if "acc_x" in col if "baseline" not in r["name"]] 
maj_y = [r["maj"][col] for r in results for col in r["maj"] if "acc_y" in col if "baseline" not in r["name"]] 
maj_z = [r["maj"][col] for r in results for col in r["maj"] if "acc_z" in col if "baseline" not in r["name"]] 

baseline_x = [r["maj"][col] for r in results for col in r["maj"] if "acc_x" in col if "baseline" in r["name"]][0] 
baseline_y = [r["maj"][col] for r in results for col in r["maj"] if "acc_y" in col if "baseline" in r["name"]][0] 
baseline_z = [r["maj"][col] for r in results for col in r["maj"] if "acc_z" in col if "baseline" in r["name"]][0] 

deltas_x = [v - baseline_x for v in maj_x]
deltas_y = [v - baseline_y for v in maj_y]
deltas_z = [v - baseline_z for v in maj_z]

# fig, axes = plt.subplots(3, 1, figsize=(16, 8))
fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)


axes[0].barh(exp_names, deltas_x, left=baseline_x, color="gray")
axes[0].axvline(baseline_x, color='red', linewidth=3.5, linestyle='-')
axes[0].set_ylabel("|X axis mean jerk|", rotation=-90)

axes[1].barh(exp_names, deltas_y, left=baseline_y, color="gray")
axes[1].axvline(baseline_y, color='red', linewidth=3.5, linestyle='-')
axes[1].set_ylabel("|Y axis mean jerk|", rotation=-90)

axes[2].barh(exp_names, deltas_z, left=baseline_z, color="gray")
axes[2].axvline(baseline_z, color='red', linewidth=3.5, linestyle='-')
axes[2].set_ylabel("|Z axis mean jerk|", rotation=-90)

# Y labels
for ax in axes:
    ax.yaxis.set_label_position("right")
    ax.yaxis.labelpad = 15

# shared x axis
all_values = maj_x + maj_y + maj_z + [baseline_x] + [baseline_y] + [baseline_z]
x_min = min(all_values) 
x_max = max(all_values) 

axes[0].set_xlim(x_min, x_max)

# Formatting
fig.supylabel('Experiments')
fig.suptitle('Mean Absolute Jerk (m/s^3) for all three axis')

plt.show()