import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pandas.core.dtypes.common import is_float_dtype
from mobgap.data_transform import HampelFilter
from mobgap.data_transform.base import BaseFilter


def average_over_intervals(
    sample_positions: np.ndarray,
    values: np.ndarray,
    interval_bounds: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean of all values located within each interval range.

    Notes
    -----
    This assumes `sample_positions` is sorted in ascending order.
    Values on interval boundaries are included.
    NaN values are ignored using `nanmean`.
    If no data falls in an interval, the output will be NaN.

    Parameters
    ----------
    sample_positions : array
        Positions of measurements along the time or sample axis.
    values : array
        Measurement values corresponding to `sample_positions`.
    interval_bounds : array
        A (N x 2) array specifying start and end of each interval.

    Returns
    -------
    array
        Mean values per interval.
    """
    n_intervals = len(interval_bounds)
    if len(sample_positions) == 0:
        return np.full(n_intervals, np.nan)
    if n_intervals == 0:
        return np.empty(0)

    # Determine positions that fall into each interval
    starts = np.searchsorted(sample_positions, interval_bounds[:, 0], side="left")
    ends = np.searchsorted(sample_positions, interval_bounds[:, 1], side="right")

    out = np.empty(n_intervals)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for i, (s, e) in enumerate(zip(starts, ends)):
            out[i] = np.nanmean(values[s:e])
    return out


def interpolate_step_metric(
    ic_times_sec: np.ndarray,
    step_metric: np.ndarray,
    second_centers: np.ndarray,
    max_gap_s: float,
    filter_obj: BaseFilter = HampelFilter(2, 3.0),
) -> np.ndarray:
    """
    Convert a per-step metric into a per-second time series by smoothing and interpolation.

    Processing flow:
    1) Filter the step-level signal.
    2) Transform to a per-second sequence by interval averaging.
    3) Smooth again.
    4) Linearly interpolate interior gaps, ignoring large breaks.

    Parameters
    ----------
    ic_times_sec : array
        Times of initial contacts (seconds).
    step_metric : array
        One value per step, aligned with initial contacts.
    second_centers : array
        Sequence of per-second center times.
    max_gap_s : float
        Maximum time window for interpolation gaps.
    filter_obj : BaseFilter
        Filtering instance applied twice.

    Returns
    -------
    array
        Interpolated per-second values.
    """
    if len(ic_times_sec) != len(step_metric):
        raise ValueError("`ic_times_sec` and `step_metric` must have equal length.")
    if len(ic_times_sec) == 0:
        return np.full(len(second_centers), np.nan)

    # Step-level smoothing
    smoothed_step = filter_obj.clone().filter(step_metric).transformed_data_

    # Interval averaging per second
    half_window = 0.5
    bounds = np.column_stack(
        (second_centers - half_window, second_centers + half_window)
    )
    per_second = average_over_intervals(ic_times_sec, smoothed_step, bounds)

    # Apply smoothing at second level
    second_smoothed = pd.Series(filter_obj.filter(per_second).transformed_data_)

    # Identify continuous missing segments
    mask = second_smoothed.isna()
    group_ids = (~mask).cumsum()
    group_lengths = mask.groupby(group_ids).transform("sum")
    # group_lengths contains lengths only for NaN segments

    # Interpolate internal gaps
    interpolated = second_smoothed.interpolate(
        method="linear", limit_area="inside"
    ).where(group_lengths <= max_gap_s)

    return interpolated.to_numpy()


def map_seconds_to_regions(
    regions: pd.DataFrame,
    second_params: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """
    Aggregate per-second metrics over regions defined by sample start/stop.

    Uses cumulative integration of second-level values and linear interpolation
    to obtain region-wise averages, independent of region alignment to the second grid.

    Parameters
    ----------
    regions : DataFrame
        Must contain 'start' and 'end' columns.
    second_params : DataFrame
        Indexed by 'sec_center_samples', each row corresponding to one second.
    sampling_rate_hz : float
        Conversion factor for sample indexing.

    Returns
    -------
    DataFrame
        Regions with additional columns representing averaged second metrics.
    """
    if regions.empty:
        return regions.reindex(
            columns=list(regions.columns) + list(second_params.columns)
        )

    if second_params.empty:
        empty_block = pd.DataFrame(index=regions.index, columns=second_params.columns)
        return pd.concat([regions, empty_block], axis=1)

    invalid_cols = [
        col for col, dt in second_params.dtypes.items() if not is_float_dtype(dt)
    ]
    if invalid_cols:
        raise ValueError(
            f"Non-float columns found in second_params: {invalid_cols}. "
            "Convert to floats before region aggregation."
        )

    sec_positions = second_params.index.get_level_values(
        "sec_center_samples"
    ).to_numpy()
    values = second_params.to_numpy()

    # Convert from second-center to second-end for integration logic
    shifted_positions = sec_positions + sampling_rate_hz * 0.5

    # Left padding to ensure interpolation validity
    pad_val = shifted_positions[0] - sampling_rate_hz
    padded_positions = np.concatenate(([pad_val], shifted_positions))
    padded_values = np.vstack((values[0], values))

    cumulative = np.cumsum(padded_values, axis=0) * sampling_rate_hz
    interpolation_fn = interp1d(
        padded_positions, cumulative, axis=0, fill_value="extrapolate"
    )

    starts = regions["start"].to_numpy()
    ends = regions["end"].to_numpy()
    stacked = np.vstack((starts, ends))
    integrated = interpolation_fn(stacked)

    durations = (ends - starts)[:, None]
    means = (integrated[1] - integrated[0]) / durations

    out = pd.DataFrame(
        means, columns=second_params.columns, index=regions.index
    ).astype(second_params.dtypes)
    return pd.concat([regions, out], axis=1)
