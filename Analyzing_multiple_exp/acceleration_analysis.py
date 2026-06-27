from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


FILES = [
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/front/adj_single_tier_front_8kg_NPA_loop3_imu_20260320_125855.csv", "exp_name": "Adj_front_8kg", "start": 500*3, "end": 10500*3},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/center/adj_single_tier_center_8kg_NPA_loop3_imu_20260320_132519.csv", "exp_name": "Adj_center_8kg", "start": 2000*3, "end": 10700*3},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_center/double_tier_center_8kg_NPA_loop3_imu_20260320_162856.csv", "exp_name": "Stack_center_8kg", "start": 1500*3, "end": 10000*3},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_front/double_tier_front_8kg_NPA_loop3_imu_20260320_151917.csv", "exp_name": "Stack_front_8kg", "start": 2000*3, "end": 12000*3},

    # {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/front_rear/single_tier_front_rear_8kg_NPA_loop3_imu_20260320_143151.csv", "exp_name": "front_rear_8kg", "start": 1600*3, "end": 10600*3},

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_imu.csv", "exp_name": "center_crate_8kg", "start": 2000*3, "end": 10500*3},
    {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_frontcrate_m_8kg/exp1/flat_frontcrate_m_8kg_exp1_imu.csv", "exp_name": "front_crate_8kg", "start": 1500*3, "end": 11000*3},

    # {"data_file": "", "exp_name": "baseline", "start": , "end": },

    {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_imu_20260331_111335.csv", "exp_name": "baseline", "start": 11240*3, "end": 19700*3}, 

]

# FILES = [
#     # All NPA
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_imu_20260331_111335.csv", "exp_name": "baseline", "mass": 33.8}, 
   
#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_NPA/incline_flat_8kg_front_crate_NPA_imu_20260413_140451.csv", "exp_name": "Front  Crate NPA", "mass": 33.8+ 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_NPA/incline_flat_8kg_center_crate_NPA_imu_20260413_142042.csv", "exp_name": "Center Crate NPA", "mass": 33.8+ 11.6},

#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/rear_crate_NPA/incline_flat_8kg_rear_crate_NPA_imu_20260413_142659.csv", "exp_name": "Rear Crate NPA", "mass": 33.8+ 11.6},
#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_NPA/incline_flat_8kg_stack_front_crate_NPA_imu_20260413_152852.csv", "exp_name": "Stack Front NPA", "mass": 33.8+ 11.6}, 

#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_NPA/incline_flat_8kg_stack_center_crate_NPA_imu_20260413_155829.csv", "exp_name": "Stack Center NPA", "mass": 33.8+ 11.6},

#     # All PA
#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/front_crate_PA/incline_flat_8kg_front_crate_PA_imu_20260413_150510.csv", "exp_name": "Front Crate PA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/center_crate_PA/incline_flat_8kg_center_crate_PA_imu_20260413_144946.csv", "exp_name": "Center Crate PA", "mass": 33.8 + 11.6},
#     # {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_front_PA/incline_flat_8kg_stack_front_crate_PA_imu_20260413_153830.csv", "exp_name": "Stack Front PA", "mass": 33.8 + 11.6},
#     {"data_file": "/home/nerve/Desktop/data_collected/Incline_flat_Apr_13/stack_center_PA/incline_flat_8kg_stack_center_crate_PA_imu_20260413_160240.csv", "exp_name": "Stack Center PA", "mass": 33.8 + 11.6},

# ]

DOWN_SAMPLE = 20
CUTOFF_HZ = 10.0   # must be < Nyquist (fs/2); keep < fs/2 when changing DOWN_SAMPLE

CUTOFF_HZ_ANG_X = 15
CUTOFF_HZ_ANG_Y = 5
CUTOFF_HZ_ANG_Z = 1

FILTER_ORDER = 2
MAX_SAMPLES = None
ACCEL_COLS = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS  = ["gyro_x", "gyro_y", "gyro_z"]


# --- helper functions --------------------------------------------------------

def load_and_downsample(file_path, downsample, max_samples=MAX_SAMPLES, start=None, end=None):
    df = pd.read_csv(file_path)
    if "start" in file:
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


# def mean_abs_jerk(jerk_array):
#     return np.mean(np.abs(jerk_array))

def rms_jerk(jerk_array):
    return np.sqrt(np.mean(np.square(jerk_array)))


# --- processing --------------------------------------------------------------

fs = int(1000 / DOWN_SAMPLE)
b, a = make_lowpass_filter(CUTOFF_HZ, fs)

# Since angular acceleration has different frequencies
b_ang_x, a_ang_x = make_lowpass_filter(CUTOFF_HZ_ANG_X, fs)
b_ang_y, a_ang_y = make_lowpass_filter(CUTOFF_HZ_ANG_Y, fs)
b_ang_z, a_ang_z = make_lowpass_filter(CUTOFF_HZ_ANG_Z, fs)

results = []
for file in FILES:
    df = load_and_downsample(file["data_file"], DOWN_SAMPLE, start=file.get("start"), end=file.get("end"))
    filtered      = apply_filter(b, a, df, columns=ACCEL_COLS)
    
    filtered_gyro = {
        "gyro_x": filtfilt(b_ang_x, a_ang_x, df["gyro_x"].values),
        "gyro_y": filtfilt(b_ang_y, a_ang_y, df["gyro_y"].values),
        "gyro_z": filtfilt(b_ang_z, a_ang_z, df["gyro_z"].values),
    }

    jerk_linear     = {col: compute_jerk(filtered[col],      fs) for col in ACCEL_COLS}
    jerk_angular = {col: compute_jerk(filtered_gyro[col], fs) for col in GYRO_COLS}

    rms_linear_jerk      = {col: rms_jerk(jerk_linear[col]) for col in ACCEL_COLS}
    rms_angular_jerk = {col: rms_jerk(jerk_angular[col]) for col in GYRO_COLS}

    results.append({
        "name": file["exp_name"],
        "filtered": filtered,
        "jerk_linear": jerk_linear,
        "rms_linear_jerk": rms_linear_jerk,
        "jerk_angular": jerk_angular,
        "rms_angular_jerk": rms_angular_jerk,
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

# --- plot 3: diverging bar chart for RMS linear jerk with 3 subplots for x, y, and z

exp_names = [r["name"] for r in results if "baseline" not in r["name"]]

rms_lin_jerk_x = [r["rms_linear_jerk"][col] for r in results for col in r["rms_linear_jerk"] if "acc_x" in col if "baseline" not in r["name"]]
rms_lin_jerk_y = [r["rms_linear_jerk"][col] for r in results for col in r["rms_linear_jerk"] if "acc_y" in col if "baseline" not in r["name"]]
rms_lin_jerk_z = [r["rms_linear_jerk"][col] for r in results for col in r["rms_linear_jerk"] if "acc_z" in col if "baseline" not in r["name"]]

baseline_x = [r["rms_linear_jerk"][col] for r in results for col in r["rms_linear_jerk"] if "acc_x" in col if "baseline" in r["name"]][0]
baseline_y = [r["rms_linear_jerk"][col] for r in results for col in r["rms_linear_jerk"] if "acc_y" in col if "baseline" in r["name"]][0]
baseline_z = [r["rms_linear_jerk"][col] for r in results for col in r["rms_linear_jerk"] if "acc_z" in col if "baseline" in r["name"]][0]

deltas_x = [v - baseline_x for v in rms_lin_jerk_x]
deltas_y = [v - baseline_y for v in rms_lin_jerk_y]
deltas_z = [v - baseline_z for v in rms_lin_jerk_z]

fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

axes[0].barh(exp_names, deltas_x, left=baseline_x, color="gray")
axes[0].axvline(baseline_x, color='red', linewidth=3.5, linestyle='-')
axes[0].set_ylabel(r"$j_{x, RMS}$", rotation=-90)

axes[1].barh(exp_names, deltas_y, left=baseline_y, color="gray")
axes[1].axvline(baseline_y, color='red', linewidth=3.5, linestyle='-')
axes[1].set_ylabel(r"$j_{y, RMS}$", rotation=-90)

axes[2].barh(exp_names, deltas_z, left=baseline_z, color="gray")
axes[2].axvline(baseline_z, color='red', linewidth=3.5, linestyle='-')
axes[2].set_ylabel(r"$j_{z, RMS}$", rotation=-90)

for ax in axes:
    ax.yaxis.set_label_position("right")
    ax.yaxis.labelpad = 15

all_values = rms_lin_jerk_x + rms_lin_jerk_y + rms_lin_jerk_z + [baseline_x, baseline_y, baseline_z]
axes[0].set_xlim(min(all_values), max(all_values))

fig.supylabel('Experiments')
# fig.suptitle('Incline Flat [NPA vs PA]')

plt.show()


# --- plot 4: diverging bar chart for RMS angular jerk (gyro x, y, z) ------

gyro_exp_names = [r["name"] for r in results if "baseline" not in r["name"]]

rms_gx = [r["rms_angular_jerk"]["gyro_x"] for r in results if "baseline" not in r["name"]]
rms_gy = [r["rms_angular_jerk"]["gyro_y"] for r in results if "baseline" not in r["name"]]
rms_gz = [r["rms_angular_jerk"]["gyro_z"] for r in results if "baseline" not in r["name"]]

baseline_gx = next(r["rms_angular_jerk"]["gyro_x"] for r in results if "baseline" in r["name"])
baseline_gy = next(r["rms_angular_jerk"]["gyro_y"] for r in results if "baseline" in r["name"])
baseline_gz = next(r["rms_angular_jerk"]["gyro_z"] for r in results if "baseline" in r["name"])

deltas_gx = [v - baseline_gx for v in rms_gx]
deltas_gy = [v - baseline_gy for v in rms_gy]
deltas_gz = [v - baseline_gz for v in rms_gz]

fig2, axes2 = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

axes2[0].barh(gyro_exp_names, deltas_gx, left=baseline_gx, color="gray")
axes2[0].axvline(baseline_gx, color='red', linewidth=1, linestyle='-')
axes2[0].set_ylabel("X axis RMS angular jerk", rotation=-90)

axes2[1].barh(gyro_exp_names, deltas_gy, left=baseline_gy, color="gray")
axes2[1].axvline(baseline_gy, color='red', linewidth=1, linestyle='-')
axes2[1].set_ylabel("Y axis RMS angular jerk", rotation=-90)

axes2[2].barh(gyro_exp_names, deltas_gz, left=baseline_gz, color="gray")
axes2[2].axvline(baseline_gz, color='red', linewidth=1, linestyle='-')
axes2[2].set_ylabel("Z axis RMS angular jerk", rotation=-90)

for ax in axes2:
    ax.yaxis.set_label_position("right")
    ax.yaxis.labelpad = 15

all_gyro_values = rms_gx + rms_gy + rms_gz + [baseline_gx, baseline_gy, baseline_gz]
axes2[0].set_xlim(min(all_gyro_values), max(all_gyro_values))

fig2.supylabel('Experiments')
# fig2.suptitle('Incline Flat [NPA vs PA]')

plt.show()