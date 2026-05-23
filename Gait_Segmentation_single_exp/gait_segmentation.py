import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILE = "/home/nerve/Desktop/data_collected/incline_crossing_May_13/stack_front_8kg_NPA/incline_crossing_stack_front_8kg_NPA_joints_20260513_170453.csv"

# contact states: 1 = ground, 2 = air
# gait cycle: touchdown (2->1) to next touchdown

def get_cycle_starts(contact: np.ndarray) -> np.ndarray:
    transition_indices = np.flatnonzero(np.diff(contact)) + 1

    before = contact[transition_indices - 1]
    after  = contact[transition_indices]

    touchdown_mask = (before == 2) & (after == 1)
    return transition_indices[touchdown_mask]


def segment_gait_cycles(df: pd.DataFrame, leg: str = "fl", max_cycle_len: int = 2000) -> list[tuple[int, int]]:
    contact = df[f"contact_{leg}"].values
    cycle_starts = get_cycle_starts(contact)

    segments = [(cycle_starts[i], cycle_starts[i + 1]) for i in range(len(cycle_starts) - 1)]

    if max_cycle_len is not None:
        segments = [(s, e) for s, e in segments if (e - s) <= max_cycle_len]

    return segments


def map_phase(df: pd.DataFrame, segments: list[tuple[int, int]]) -> pd.DataFrame:
    phase_col = np.full(len(df), np.nan)

    for s, e in segments:
        phase_col[s:e] = np.linspace(0, 1, e - s)

    df["phase"] = phase_col
    return df


def resample_cycles(df: pd.DataFrame, segments: list[tuple[int, int]],
                    columns: list[str], n_points: int = 100) -> dict[str, np.ndarray]:
    target_phase = np.linspace(0, 1, n_points)
    resampled = {col: np.empty((len(segments), n_points)) for col in columns}

    for i, (s, e) in enumerate(segments):
        original_phase = np.linspace(0, 1, e - s)
        for col in columns:
            signal = df[col].values[s:e]
            resampled[col][i] = np.interp(target_phase, original_phase, signal)

    return resampled


def plot_signal_with_segments(df: pd.DataFrame, segments: list[tuple[int, int]], column: str) -> None:
    _, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[column].values, linewidth=0.8, color="steelblue")

    for s, _ in segments:
        ax.axvline(x=s, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(x=segments[-1][1], color="red", linewidth=0.8, linestyle="--", alpha=0.6, label="cycle boundary")

    ax.set_xlabel("Row index")
    ax.set_ylabel(column)
    ax.set_title(f"{column} with gait cycle boundaries")
    ax.legend(loc="upper right")
    plt.tight_layout()

    plt.show()


def plot_mean_std_per_phase(resampled: dict[str, np.ndarray], n_points: int = 100) -> None:
    phase   = np.linspace(0, 1, n_points)
    columns = list(resampled.keys())
    n_cols  = len(columns)

    _, axes = plt.subplots(n_cols, 1, figsize=(10, 4 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        matrix = resampled[col]
        mean   = matrix.mean(axis=0)
        std    = matrix.std(axis=0)

        ax.plot(phase, mean, linewidth=2, label="Mean")
        ax.fill_between(phase, mean - std, mean + std, alpha=0.25, label="1 standard deviation")
        ax.set_ylabel(col)
        ax.legend(loc="upper right")
        ax.grid(True)

    axes[-1].set_xlabel("Gait phase")
    plt.tight_layout()
    plt.savefig("joint_torque_phase.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_rms_per_phase(resampled: dict[str, np.ndarray], n_points: int = 100) -> None:
    phase   = np.linspace(0, 1, n_points)
    columns = list(resampled.keys())
    n_cols  = len(columns)

    _, axes = plt.subplots(n_cols, 1, figsize=(10, 4 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        matrix = resampled[col]
        rms    = np.sqrt(np.mean(matrix ** 2, axis=0))

        ax.plot(phase, rms, linewidth=2, color="darkorange", label="RMS")
        ax.set_ylabel(col)
        ax.legend(loc="upper right")
        ax.grid(True)

    axes[-1].set_xlabel("Gait phase")
    plt.suptitle("RMS per gait phase")
    plt.tight_layout()
    plt.savefig("joint_torque_rms_phase.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(FILE)

    segments = segment_gait_cycles(df)
    df = map_phase(df, segments)

    columns   = ["fl.hx_load", "fl.hy_load", "fl.kn_load"]
    resampled = resample_cycles(df, segments, columns=columns, n_points=100)

    print(f"Total gait cycles found: {len(segments)}")

    # plot_signal_with_segments(df, segments, "fl.kn_load") -> Sanity check for segmentation
    plot_rms_per_phase(resampled)
