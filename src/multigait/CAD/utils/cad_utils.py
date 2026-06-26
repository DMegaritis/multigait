import numpy as np
import pandas as pd
from mobgap.data_transform import HampelFilter
import warnings


def smooth_and_bin_steps(
    step_times: np.ndarray,
    step_values: np.ndarray,
    second_midpoints: np.ndarray,
    max_gap_s: float,
    filter_obj: HampelFilter,
) -> np.ndarray:
    """
    Smooth per-step measurements and interpolate to a per-second resolution.

    Parameters
    ----------
    step_times : np.ndarray
        Timestamps of initial contacts in seconds.
    step_values : np.ndarray
        The per-step metric to interpolate (e.g., step durations).
    second_midpoints : np.ndarray
        Time points (in seconds) for which the per-second values are computed.
    max_gap_s : float
        Maximum gap in seconds to linearly interpolate. Gaps larger than this remain NaN.
    filter_obj : HampelFilter
        Hampel filter used to remove outliers. Applied before and after interpolation.

    Returns
    -------
    np.ndarray
        Per-second interpolated values, after smoothing and gap-filling.
    """
    if len(step_times) != len(step_values):
        raise ValueError("Step times and step values must have the same length.")
    if len(step_times) == 0:
        return np.full(len(second_midpoints), np.nan)

        # Smooth per-step values
    smoothed_steps = filter_obj.clone().filter(step_values).transformed_data_

    # Average over each second
    intervals = np.vstack([second_midpoints - 0.5, second_midpoints + 0.5]).T
    binned_steps = compute_interval_mean(step_times, smoothed_steps, intervals)

    # Apply second smoothing
    smoothed_per_sec = pd.Series(filter_obj.filter(binned_steps).transformed_data_)

    # Gap-aware linear interpolation
    valid_mask = smoothed_per_sec.notna()
    gap_groups = valid_mask.ne(valid_mask.shift()).cumsum()
    gap_sizes = (
        smoothed_per_sec.groupby([gap_groups, smoothed_per_sec.isna()])
        .transform("size")
        .where(smoothed_per_sec.isna())
    )
    smoothed_per_sec = smoothed_per_sec.interpolate(
        method="linear", limit_area="inside"
    ).mask(gap_sizes > max_gap_s)

    return smoothed_per_sec.to_numpy()


def compute_interval_mean(
    timestamps: np.ndarray,
    values: np.ndarray,
    interval_bounds: np.ndarray,
) -> np.ndarray:
    """
    Compute the average of measurements within specified intervals.

    Each interval includes all measurements between the start and end bounds.
    NaN values in `values` are ignored. If no measurements exist in an interval, the result is NaN.

    Parameters
    ----------
    timestamps : np.ndarray
        Times at which each measurement occurs (sorted in ascending order).
    values : np.ndarray
        Measurements corresponding to the timestamps.
    interval_bounds : np.ndarray
        Array of shape (n_intervals, 2) with start and end times of each interval.

    Returns
    -------
    np.ndarray
        Mean value for each interval.
    """

    if len(timestamps) == 0:
        return np.full(len(interval_bounds), np.nan)
    if len(interval_bounds) == 0:
        return np.array([])

    starts = np.searchsorted(timestamps, interval_bounds[:, 0], side="left")
    ends = np.searchsorted(timestamps, interval_bounds[:, 1], side="right")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.array([np.nanmean(values[s:e]) for s, e in zip(starts, ends)])

    return means
