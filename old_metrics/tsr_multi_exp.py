"""
Compare Torque Symmetry Ratio (TSR) across multiple load-placement experiments.

For each condition, one experiment trial is selected. Per-cycle TSR values are
plotted as scatter+line in a 2×3 grid — one subplot per joint pair, one line
per condition.

Gait cycle segmentation uses FL foot touch-down events (contact_fl: 2→1).

Usage:
    python3 tsr_multi_exp.py <parent_data_dir> [--exp N] [--conditions C1 C2 ...]

Example:
    python3 tsr_multi_exp.py /home/nerve/Desktop/data_collected/flat_Feb21
    python3 tsr_multi_exp.py /home/nerve/Desktop/data_collected/flat_Feb4 --exp 2
    python3 tsr_multi_exp.py /home/nerve/Desktop/data_collected/flat_Feb21 --conditions flat_centercrate_m_8kg flat_frontcrate_m_8kg flat_rearcrate_m_8kg flat_centercrate_m_8kg_payload_aware flat_frontcrate_m_8kg_payload_aware
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRIM_START_SEC = 7.0   # seconds to skip from the start of each recording
MAX_CYCLES     = 40    # only plot the first N gait cycles

# ── Legend display names ──────────────────────────────────────────────────────
# Map condition folder names to shorter display names for the legend.
# Any condition not listed here will use its folder name as-is.
LEGEND_NAMES = {
    'flat_centercrate_b_8kg': 'Center (back)',
    'flat_centercrate_f_8kg': 'Center (front)',
    'flat_centercrate_m_8kg': 'Center payload 8kg',
    'flat_frontcrate_b_8kg':  'Front (back)',
    'flat_frontcrate_f_8kg':  'Front (front)',
    'flat_frontcrate_m_8kg':  'Front payload 8kg',
    'flat_rearcrate_f_8kg':   'Rear (front)',

    # payload aware
    'flat_centercrate_m_8kg_payload_aware': 'Center 8kg: Payload aware',
    'flat_frontcrate_m_8kg_payload_aware': 'Front 8kg: Payload aware',

}

# ── Joint pairs ───────────────────────────────────────────────────────────────
JOINT_PAIRS = [
    ('fl.hx_load', 'fr.hx_load', 'Front HX'),
    ('fl.hy_load', 'fr.hy_load', 'Front HY'),
    ('fl.kn_load', 'fr.kn_load', 'Front Knee'),
    ('hl.hx_load', 'hr.hx_load', 'Hind HX'),
    ('hl.hy_load', 'hr.hy_load', 'Hind HY'),
    ('hl.kn_load', 'hr.kn_load', 'Hind Knee'),
]


# ── Data discovery ────────────────────────────────────────────────────────────

def find_joints_csv(parent_dir, condition, exp_idx):
    exp_name = f'exp{exp_idx}'
    # filename = f'{condition}_{exp_name}_joints.csv'
    filename = f'{condition}_{exp_name}.csv'

    return os.path.join(parent_dir, condition, exp_name, filename)


def discover_conditions(parent_dir, exp_idx):
    found = []
    for entry in sorted(os.listdir(parent_dir)):
        full = os.path.join(parent_dir, entry)
        if not os.path.isdir(full):
            continue
        csv_path = find_joints_csv(parent_dir, entry, exp_idx)
        if os.path.isfile(csv_path):
            found.append((entry, csv_path))
        else:
            print(f'  [skip] {entry}: no exp{exp_idx} joints CSV found')
    return found


# ── Core computation ──────────────────────────────────────────────────────────

def detect_fl_touchdowns(contact_fl):
    c = np.array(contact_fl)
    return np.where((c[:-1] == 2) & (c[1:] == 1))[0] + 1


def compute_tsr_for_file(csv_path):
    """Compute per-cycle TSR and peak torques for all joint pairs from one joints CSV.

    Returns:
        cycle_numbers : (n_cycles,) array [1, 2, 3, ...]
        tsr_results   : dict mapping label -> (n_cycles,) array of TSR values
        peak_results  : dict mapping label -> {'left': (n_cycles,), 'right': (n_cycles,)}
    """
    df = pd.read_csv(csv_path)
    df = df[df['elapsed_time'] >= TRIM_START_SEC].reset_index(drop=True)
    df['elapsed_time'] = df['elapsed_time'] - TRIM_START_SEC

    touchdowns = detect_fl_touchdowns(df['contact_fl'].values)
    if len(touchdowns) < 2:
        raise ValueError(f'Not enough gait cycles in {csv_path}')

    n_cycles = min(len(touchdowns) - 1, MAX_CYCLES)
    cycle_numbers = np.arange(1, n_cycles + 1)

    tsr_results  = {}
    peak_results = {}
    for left_col, right_col, label in JOINT_PAIRS:
        left  = df[left_col].values
        right = df[right_col].values
        cycle_tsr   = []
        peaks_left  = []
        peaks_right = []
        for i in range(n_cycles):
            s, e = touchdowns[i], touchdowns[i + 1]
            peak_left  = np.max(np.abs(left[s:e]))
            peak_right = np.max(np.abs(right[s:e]))
            cycle_tsr.append(peak_left / peak_right if peak_right > 0 else np.nan)
            peaks_left.append(peak_left)
            peaks_right.append(peak_right)
        tsr_results[label]  = np.array(cycle_tsr)
        peak_results[label] = {
            'left':  np.array(peaks_left),
            'right': np.array(peaks_right),
        }

    return cycle_numbers, tsr_results, peak_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tsr_trajectories(conditions, all_results):
    """Plot per-cycle TSR for all conditions in a 2×3 grid.

    Each subplot = one joint pair.
    Each line = one condition.
    """
    labels     = [label for _, _, label in JOINT_PAIRS]
    cond_names = [name for name, _ in conditions]
    colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]

        for c_idx, name in enumerate(cond_names):
            cycle_numbers, tsr_results, _ = all_results[name]
            tsr   = tsr_results[label]
            color = colors[c_idx % len(colors)]

            ax.plot(cycle_numbers, tsr, color=color, linewidth=0.8, zorder=1)
            ax.scatter(cycle_numbers, tsr, s=18, color=color, zorder=2)

        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(False)

    # Single legend outside the subplots
    handles = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=1.5,
                   marker='o', markersize=4, label=LEGEND_NAMES.get(name, name))
        for i, name in enumerate(cond_names)
    ]
    fig.legend(handles=handles, loc='upper right', ncol=1, fontsize=10, bbox_to_anchor=(0.98, 0.90))



    # fig.suptitle('TSR per Gait Cycle  (left / right peak torque)\n1.0 = perfect symmetry',fontsize=12)
    fig.supxlabel('Gait cycle', fontsize=10)
    fig.supylabel('TSR', fontsize=10)
    plt.tight_layout()
    plt.show()


# ── Peak torque summary table ─────────────────────────────────────────────────

def print_peak_torque_table(conditions, all_results):
    """Print mean ± SD of per-cycle peak torques for all 12 joints in one row per condition.

    Column names are derived from the CSV column names (e.g. fl.hx_load → FL.HX).
    """
    cond_names = [name for name, _ in conditions]

    def col_label(col):
        return col.replace('_load', '').upper()   # fl.hx_load → FL.HX

    # Build ordered list of (display_name, side) for 12 columns
    columns = []
    for left_col, right_col, _ in JOINT_PAIRS:
        columns.append((col_label(left_col),  'left'))
        columns.append((col_label(right_col), 'right'))

    col_w = 12   # width for each "mean±SD" cell

    print('\n── Peak Torque (Nm)  —  mean ± SD across gait cycles ──')

    header = f'  {"Condition":<26}'
    for disp, _ in columns:
        header += f'  {disp:^{col_w}}'
    print(header)
    print('  ' + '-' * (26 + len(columns) * (col_w + 2)))

    for name in cond_names:
        _, _, peak_results = all_results[name]
        display = LEGEND_NAMES.get(name, name)
        row = f'  {display:<26}'
        for _, _, label in JOINT_PAIRS:
            peaks = peak_results[label]
            ml, sl = np.nanmean(peaks['left']),  np.nanstd(peaks['left'])
            mr, sr = np.nanmean(peaks['right']), np.nanstd(peaks['right'])
            row += f'  {ml:>4.1f}±{sl:<4.1f}  {mr:>4.1f}±{sr:<4.1f}'
        print(row)


# ── Peak torque comparison plot ───────────────────────────────────────────────

def plot_peak_torque_comparison(conditions, all_results):
    """2×3 errorbar grid: mean ± SD of per-cycle peak torque per joint per condition.

    Each subplot = one joint pair.
    Each condition gets two errorbars (slightly offset): left joint and right joint.
    Mean and SD are computed across gait cycles within the experiment.
    """
    labels     = [label for _, _, label in JOINT_PAIRS]
    cond_names = [name for name, _ in conditions]
    colors     = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_pos   = np.arange(len(cond_names))
    offset  = 0.15   # horizontal offset between left and right errorbars

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]

        for c_idx, name in enumerate(cond_names):
            _, _, peak_results = all_results[name]
            peaks = peak_results[label]
            color = colors[c_idx % len(colors)]

            mean_l, sd_l = np.nanmean(peaks['left']),  np.nanstd(peaks['left'])
            mean_r, sd_r = np.nanmean(peaks['right']), np.nanstd(peaks['right'])

            ax.errorbar(c_idx - offset, mean_l, yerr=sd_l,
                        fmt='o', color=color, markersize=5,
                        capsize=3, capthick=1, elinewidth=1,
                        label='Left' if c_idx == 0 else '')
            ax.errorbar(c_idx + offset, mean_r, yerr=sd_r,
                        fmt='s', color=color, markersize=5,
                        capsize=3, capthick=1, elinewidth=1, alpha=0.5,
                        label='Right' if c_idx == 0 else '')

        ax.set_title(label, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [LEGEND_NAMES.get(n, n.replace('_', ' ')) for n in cond_names],
            fontsize=8, rotation=20, ha='right')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(False)
        if idx in (0, 3):
            ax.set_ylabel('Peak torque (Nm)', fontsize=9)

    # Legend: circle = left, square = right
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                   markersize=6, label='Left'),
        plt.Line2D([0], [0], marker='s', color='gray', linestyle='None',
                   markersize=6, alpha=0.5, label='Right'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=9,
               bbox_to_anchor=(0.98, 0.96))

    fig.supxlabel('Condition', fontsize=10)
    fig.supylabel('Peak torque (Nm)', fontsize=10)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot per-cycle TSR trajectories across load-placement experiments')
    parser.add_argument('parent_dir',
                        help='Parent folder containing one subfolder per condition')
    parser.add_argument('--exp', type=int, default=1,
                        help='Which exp index to use from each condition (default: 1)')
    parser.add_argument('--conditions', nargs='+', default=None, metavar='COND',
                        help='Condition names to include (default: all discovered)')
    args = parser.parse_args()

    print(f'Scanning: {args.parent_dir}  (exp{args.exp})\n')
    conditions = discover_conditions(args.parent_dir, args.exp)

    if args.conditions is not None:
        requested = set(args.conditions)
        found_names = {name for name, _ in conditions}
        missing = requested - found_names
        if missing:
            print(f'[WARNING] conditions not found: {", ".join(sorted(missing))}')
        conditions = [(name, csv) for name, csv in conditions if name in requested]

    if not conditions:
        print('No valid experiments found. Check parent_dir, --exp index, and --conditions.')
        sys.exit(1)

    print(f'\nFound {len(conditions)} conditions:')
    for name, _ in conditions:
        print(f'  {name}')

    all_results = {}
    print()
    for name, csv_path in conditions:
        print(f'Processing: {name}')
        try:
            cycle_numbers, tsr_results, peak_results = compute_tsr_for_file(csv_path)
            print(f'  → {len(cycle_numbers)} gait cycles')
            all_results[name] = (cycle_numbers, tsr_results, peak_results)
        except Exception as e:
            print(f'  [ERROR] {e}')
            sys.exit(1)

    print_peak_torque_table(conditions, all_results)
    # plot_tsr_trajectories(conditions, all_results)
    # plot_peak_torque_comparison(conditions, all_results)


if __name__ == '__main__':
    main()
