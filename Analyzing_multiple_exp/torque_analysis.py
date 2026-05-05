import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# FILES = [
   
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/front/adj_single_tier_front_8kg_NPA_loop3_joints_20260320_125855.csv", "exp_name": "Adj_front_8kg", "start": 500, "end": 10500},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/Adjacent/center/adj_single_tier_center_8kg_NPA_loop3_joints_20260320_132519.csv", "exp_name": "Adj_center_8kg", "start": 2000, "end": 10700},

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_center/double_tier_center_8kg_NPA_loop3_joints_20260320_162856.csv", "exp_name": "Stack_center_8kg", "start": 1500, "end": 10000},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/stacking/stack_front/double_tier_front_8kg_NPA_loop3_joints_20260320_151917.csv", "exp_name": "Stack_front_8kg", "start": 2000, "end": 12000},

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/front_rear/single_tier_front_rear_8kg_NPA_loop3_joints_20260320_143151.csv", "exp_name": "front_rear_8kg", "start": 1600, "end": 10600},

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_centercrate_m_8kg/exp1/flat_centercrate_m_8kg_exp1_joints.csv", "exp_name": "center_crate_8kg", "start": 2000, "end": 10500},
#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Feb21/flat_frontcrate_m_8kg/exp1/flat_frontcrate_m_8kg_exp1_joints.csv", "exp_name": "front_crate_8kg", "start": 1500, "end": 11000},

#     {"data_file": "/home/nerve/Desktop/data_collected/flat_Mar_20/baseline_without_rail/baseline_loop3_joints_20260331_111335.csv", "exp_name": "baseline", "start": 11240, "end": 19700}, 
# ]

FILES = [
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/baseline_incline_flat/incline_flat_baseline_joints_20260426_172759.csv", "exp_name": "baseline", "mass": 33.8}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_8kg_PA/incline_flat_8kg_adj_center_PA_joints_20260426_150208.csv", "exp_name": "Adjacent_center_8kg", "mass": 33.8 + 8.}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_12kg_PA/incline_flat_12kg_adj_center_PA_joints_20260426_160040.csv", "exp_name": "Adjacent_center_12kg", "mass": 33.8 + 12.}, 
    {"data_file": "/home/nerve/Desktop/data_collected/incline_flat_Apr_26/adj_center_14kg_PA/incline_flat_14kg_adj_center_PA_joints_20260426_171228.csv", "exp_name": "Adjacent_center_14kg", "mass": 33.8 + 14.}, 

]

DOWN_SAMPLE    = 10       # joints data is already at lower frequency than IMU
JOINTS_FS      = 333     # Hz
REFERENCE_FOOT = "fl"   # foot used to define gait cycle boundaries

LEGS   = ["fl", "fr", "hl", "hr"]
JOINTS = ["hx", "hy", "kn"]
# JOINTS = ["kn"]

TORQUE_COLS  = [f"{leg}.{joint}_load"    for leg in LEGS for joint in JOINTS]
CONTACT_COLS = [f"contact_{leg}"         for leg in LEGS]

STANCE_VALUE = 1   # 1 = foot on ground, 2 = foot in air


# --- helper functions --------------------------------------------------------

def load_and_downsample(file_path, downsample, max_samples=None, start=None, end=None):
    df = pd.read_csv(file_path)
    if start is not None or end is not None:
        df = df.iloc[start:end]
    df = df[::downsample]
    if max_samples is not None:
        df = df[:max_samples]
    return df.reset_index(drop=True)


def detect_gait_cycles(contact_series, stance_value=STANCE_VALUE):
    """Return indices of touchdowns (swing->stance transitions) for one foot."""
    contact = contact_series.values
    touchdowns = np.where(
        (contact[:-1] != stance_value) & (contact[1:] == stance_value)
    )[0] + 1
    return touchdowns


def extract_peak_torques(df, cycle_boundaries, torque_cols=TORQUE_COLS):
   
    peaks = {col: [] for col in torque_cols}
    for i in range(len(cycle_boundaries) - 1):
        start, end = cycle_boundaries[i], cycle_boundaries[i + 1]
        segment = df.iloc[start:end]
        for col in torque_cols:
            peaks[col].append(np.max(np.abs(segment[col].values)))
    return peaks


def mean_peak_torque(peaks):
    
    return {col: np.mean(vals) for col, vals in peaks.items()}

def rms_peak_torque(peaks):
    return {col: np.sqrt(np.mean(np.square(val))) for col, val in peaks.items()}

def verify_cycles_detected(df, cycle_boundaries, exp_name):
    joint_torque_FL_Knee = df[f"{REFERENCE_FOOT}.kn_load"].values
    exp_time_data = np.arange(len(joint_torque_FL_Knee))

    peak_indices = []
    peak_values = []
    for i in range(len(cycle_boundaries) - 1):
        start, end = cycle_boundaries[i], cycle_boundaries[i + 1]
        idx = start + np.argmax(np.abs(joint_torque_FL_Knee[start:end]))
        peak_indices.append(idx)
        peak_values.append(joint_torque_FL_Knee[idx])

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(exp_time_data, joint_torque_FL_Knee)
    ax.plot(peak_indices, peak_values, marker='o', color='red')
    ax.set_title(exp_name)
    plt.show()


# --- processing --------------------------------------------------------------

results = []
for file in FILES:
    if "start" in file:
        df = load_and_downsample(file["data_file"], DOWN_SAMPLE, start=file.get("start"), end=file.get("end"))
    else:
        df = load_and_downsample(file["data_file"], DOWN_SAMPLE, start=None, end=None)


    contact_col = f"contact_{REFERENCE_FOOT}"
    cycle_boundaries = detect_gait_cycles(df[contact_col])
    n_cycles = len(cycle_boundaries) - 1
    print(f"{file['exp_name']}: {n_cycles} gait cycles detected")

    peaks = extract_peak_torques(df, cycle_boundaries)
    # verify_cycles_detected(df, cycle_boundaries, file["exp_name"])
    # mpt   = mean_peak_torque(peaks)
    rms_pt = rms_peak_torque(peaks)
    # calculation for my new defined metric balance_score sigma/mean for all 4 knee joint mean torque
    # mean_for_knee_joint_torques = [vals for col, vals in mpt.items() if "kn" in col]
    # mean = np.mean(mean_for_knee_joint_torques)
    # std = np.std(mean_for_knee_joint_torques)
    # balance_score = (std / mean) * 100

    results.append({
        "name": file["exp_name"],
        "peaks": peaks,
        "rms_pt": rms_pt,
        "n_cycles": n_cycles,
        # "balance_score": balance_score,
    })


# --- plot 1: peak torque per gait cycle (one figure per experiment) ----------

# for result in results:
#     fig, axes = plt.subplots(len(LEGS), len(JOINTS), figsize=(14, 10), sharey=False)
#     fig.suptitle(f"Peak Torque per Gait Cycle — {result['name']}")

#     for r, leg in enumerate(LEGS):
#         for c, joint in enumerate(JOINTS):
#             col = f"{leg}.{joint}_load"
#             cycle_indices = np.arange(len(result["peaks"][col]))
#             axes[r, c].plot(cycle_indices, result["peaks"][col], marker='o', markersize=3)
#             axes[r, c].set_title(f"{leg}.{joint}")
#             if c == 0:
#                 axes[r, c].set_ylabel("Peak |Torque| (Nm)")
#             if r == len(LEGS) - 1:
#                 axes[r, c].set_xlabel("Gait Cycle #")

#     plt.tight_layout()


# --- plot 2: mean peak torque comparison across experiments ------------------

if len(results) > 1:
    exp_names = [r["name"] for r in results]
    n_exp = len(exp_names)
    x = np.arange(n_exp)
    width = 0.06
    n_cols = len(TORQUE_COLS)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(TORQUE_COLS):
        rms_pt_values = [r["rms_pt"][col] for r in results]
        offset = (i - n_cols / 2) * width
        ax.bar(x + offset, rms_pt_values, width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15, ha='right')
    ax.set_ylabel("RMS Peak Torque (Nm)")
    ax.set_title("RMS Peak Torque Comparison Across Experiments")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()

# --- plot 3: Diverging bar chart for torque comparision with baseline for each joint

# if len(results) > 1:
#     exp_names = [r["name"] for r in results]
#     n_exp = len(exp_names)
#     x = np.arange(n_exp)
#     width = 0.06
#     n_cols = len(TORQUE_COLS)

#     for i, col in enumerate(TORQUE_COLS):
#         # mpt_values = [r["mpt"][col] for r in results]
#         # offset = (i - n_cols / 2) * width
#         # ax.bar(x + offset, mpt_values, width, label=col)

#         fig, ax = plt.subplots(figsize=(14, 6))
#         mpt_values = [r["mpt"][col] for r in results if "baseline" not in r["name"]]
#         baseline_mpt_value = [r["mpt"][col] for r in results if "baseline" in r["name"]][0]
#         deltas = [v - baseline_mpt_value for v in mpt_values]

#         exp_names_no_baseline = [r["name"] for r in results if "baseline" not in r["name"]]
#         ax.barh(exp_names_no_baseline, deltas, left=baseline_mpt_value, color=['green' if d > 0 else 'red' for d in deltas])
#         ax.axvline(baseline_mpt_value, color='black', linewidth=1.5, linestyle='-', label=f'Baseline ({baseline_mpt_value:.2f} Nm)')

#         for y, (val, delta) in enumerate(zip(mpt_values, deltas)):
#             x_pos = val + (0.01 * abs(val)) * (1 if delta >= 0 else -1)
#             ha = 'left' if delta >= 0 else 'right'
#             ax.text(x_pos, y, f'{val:.2f}', va='center', ha=ha, fontsize=9)

#         ax.set_xlabel("Mean Peak Torque (Nm)")
#         ax.set_title(f"Torque vs Baseline — {col}")
#         ax.legend()
#         plt.tight_layout()
#         plt.show()

# --- table: mean peak torque per joint for each experiment ------------------

# table_data = {r["name"]: [round(r["mpt"][col], 2) for col in TORQUE_COLS] for r in results}
# summary_df = pd.DataFrame(table_data, index=TORQUE_COLS)
# print("\nMean Peak Torque (Nm)")
# print(summary_df.to_string())

plt.show()

# print(TORQUE_COLS)