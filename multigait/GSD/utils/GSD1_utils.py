import pandas as pd
import numba
import numpy as np
from intervaltree import IntervalTree
from numba import njit
from scipy.signal import find_peaks, hilbert
from multigait.utils.array import (
    merge_interval,
    bool_array_to_start_end,
    start_end_array_to_bool,
)


def active_regions_from_hilbert_envelop(
    sig: np.ndarray, smooth_window: int, duration: int
) -> np.ndarray:
    """Detect periods of activity using a Hilbert transform and adaptive threshold.

    Computes the analytical signal via the Hilbert transform, smooths the resulting envelope,
    and identifies active periods where the signal stays above a dynamic threshold for at least
    `duration` samples. The threshold adapts online based on signal noise and activity level.

    Parameters
    ----------
    sig : np.ndarray
        1D array representing the input signal.
    smooth_window : int
        Window length in samples used for smoothing the signal envelope.
    duration : int
        Minimum number of consecutive samples above the threshold to classify as active.

    Returns
    -------
    np.ndarray
        Boolean array of the same length as `sig`. True indicates detected activity periods.

    """

    # Calculate the analytical signal and get the envelope
    amplitude_envelope = np.abs(hilbert(sig))

    # Take the moving average of analytical signal
    env = np.convolve(
        amplitude_envelope,
        np.ones(smooth_window) / smooth_window,
        "same",  # Smooth
    )

    active = np.zeros(len(env))

    env -= np.mean(env)  # Get rid of offset
    if np.all(env == 0):
        return active.astype(bool)
    env /= np.max(env)  # Normalize

    threshold_sig = 4 * np.nanmean(env)
    noise = np.mean(env) / 3  # Noise level
    threshold = np.mean(env)  # Signal level
    update_threshold = False

    # Initialize Buffers
    noise_buff = np.zeros(len(env) - duration + 1)

    if np.isnan(threshold_sig):
        return active.astype(bool)

    maxenv = max(env)
    for i in range(len(env) - duration + 1):
        # Update threshold 10% of the maximum peaks found
        window = env[i : i + duration]
        mean_win = np.mean(window)
        if (window > threshold_sig).all():
            active[i] = maxenv
            threshold = 0.1 * mean_win
            update_threshold = True
        elif mean_win < threshold_sig:
            noise = mean_win
        elif noise_buff.any():
            noise = np.mean(noise_buff)
        # NOTE: no else case in the original implementation

        noise_buff[i] = noise

        # Update threshold
        if update_threshold:
            threshold_sig = noise + 0.50 * (abs(threshold - noise))

    return active.astype(bool)


@njit(cache=True)
def _find_pulse_train_end(x: np.ndarray, step_threshold: float) -> np.ndarray:
    start_val = x[0]

    for ic_idx, (current_val, next_val) in enumerate(zip(x[1:], x[2:])):
        # We already know that the first two values belong to the pulse train, as this is determined by the caller so we
        # start everything at index 1
        n_steps = ic_idx + 1
        # We update the threshold to be the mean step time + the step threshold
        # Note: The original implementation uses effectively n_steps + 1 here, which likely a bug, as it counts the
        # number of pulses within the pulse train and not the number of distances between pulses.
        thd_step = (current_val - start_val) / n_steps + step_threshold
        if next_val - current_val > thd_step:
            return x[: n_steps + 1]
    return x


@njit(cache=True)
def find_pulse_trains(
    x: np.ndarray,
    initial_distance_threshold_samples: float,
    step_threshold_margin: float,
) -> np.ndarray:
    start_ends = []
    i = 0
    while i < len(x) - 1:
        # We search for a start of a pulse train
        # This happens, in case 2 consecutive samples are closer than the initial distance threshold
        if x[i + 1] - x[i] < initial_distance_threshold_samples:
            # Then we search for the end of the pulse train
            # This happens, in case 2 consecutive samples are further apart than the step threshold + the mean step time
            # within the pulse train
            start = x[i]
            pulses = _find_pulse_train_end(x[i:], step_threshold_margin)
            start_ends.append([start, pulses[-1]])
            i += len(pulses)
        else:
            i += 1

    if len(start_ends) == 0:
        return np.empty((0, 2), dtype=np.int32)

    start_ends_array = np.array(start_ends, dtype=np.int32)
    return start_ends_array


def find_intersections(intervals_a: np.ndarray, intervals_b: np.ndarray) -> np.ndarray:
    """Compute intersections between two sets of intervals.

    Finds overlapping intervals between two arrays of [start, end] values.

    Parameters
    ----------
    intervals_a : np.ndarray
        First set of intervals, each represented as [start, end].
    intervals_b : np.ndarray
        Second set of intervals, each represented as [start, end].

    Returns
    -------
    np.ndarray
        Array of intervals representing overlaps between `intervals_a` and `intervals_b`.
    """
    # Create Interval Trees
    intervals_a_tree = IntervalTree.from_tuples(intervals_a)
    intervals_b_tree = IntervalTree.from_tuples(intervals_b)

    overlap_intervals = []

    for interval in intervals_b_tree:
        overlaps = sorted(intervals_a_tree.overlap(interval.begin, interval.end))
        if overlaps:
            for overlap in overlaps:
                start = max(interval.begin, overlap.begin)
                end = min(interval.end, overlap.end)
                overlap_intervals.append([start, end])

    return (
        merge_interval(np.array(overlap_intervals))
        if len(overlap_intervals) != 0
        else np.array([])
    )


class NoActivePeriodsDetectedError(Exception):
    pass


def find_active_period_peak_threshold(
    self,
    *,
    signal: np.ndarray,
    min_active_period_duration: int,
    hilbert_window_size: int,
) -> float:
    # Find pre-detection of 'active' periods in order to estimate the amplitude of acceleration peaks
    active_regions = active_regions_from_hilbert_envelop(
        signal, hilbert_window_size, hilbert_window_size
    )

    if not np.any(active_regions):
        raise NoActivePeriodsDetectedError()

    active_regions_start_end = bool_array_to_start_end(active_regions)
    to_short_active_regions = (
        active_regions_start_end[:, 1] - active_regions_start_end[:, 0]
    ) <= min_active_period_duration
    active_regions_start_end = active_regions_start_end[~to_short_active_regions]

    if len(active_regions_start_end) == 0:
        raise NoActivePeriodsDetectedError()

    final_active_area = signal[
        start_end_array_to_bool(active_regions_start_end, len(active_regions))
    ]

    _, props_p = find_peaks(final_active_area, height=0)
    _, props_n = find_peaks(-final_active_area, height=0)
    pks = np.concatenate([props_p["peak_heights"], props_n["peak_heights"]])
    return np.percentile(pks, self.percentile)  # Data adaptive threshold


def format_gait_sequences(df_or_array):
    # Handling empty input
    if isinstance(df_or_array, np.ndarray) and df_or_array.size == 0:
        df = pd.DataFrame(columns=["start", "end"])
        df.index.name = "gs_id"
        return df

    # Convert array to DataFrame if needed
    if isinstance(df_or_array, np.ndarray):
        df = pd.DataFrame(df_or_array, columns=["start", "end"])
    else:
        df = df_or_array.copy()

    # Add gs_id column if missing
    if "gs_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "gs_id"})

    # Ensure correct dtypes: gs_id=int, start/end=int
    df = df.astype({"gs_id": int, "start": int, "end": int})

    # Set gs_id as index
    return df.set_index("gs_id")


def combine_intervals(intervals: np.ndarray, max_gap: int = 0) -> np.ndarray:
    """Combine overlapping or closely spaced intervals.

    Intervals that overlap or are separated by a distance less than or equal to max_gap
    will be merged into a single interval.

    Parameters
    ----------
    intervals : np.ndarray, shape (n,2)
        Array of [start, end] intervals.
    max_gap : int
        Maximum allowed gap between intervals to merge them.

    Returns
    -------
    np.ndarray
        Array of merged intervals.
    """
    if intervals.shape[0] == 0:
        return intervals

    sorted_intervals = np.sort(intervals, axis=0, kind="stable")
    merged_list = _merge_overlap_numba(sorted_intervals, max_gap)
    return np.array(merged_list)


@njit
def _merge_overlap_numba(intervals: np.ndarray, max_gap: int) -> numba.typed.List:
    """Helper function for merging intervals using numba for speed."""
    merged = numba.typed.List()
    merged.append(intervals[0])

    for i in range(1, len(intervals)):
        current_start, current_end = intervals[i]
        last_start, last_end = merged[-1]

        if (
            last_start
            <= current_start
            <= (last_end + max_gap)
            <= (current_end + max_gap)
        ):
            merged[-1][1] = current_end
        else:
            merged.append(intervals[i])

    return merged
