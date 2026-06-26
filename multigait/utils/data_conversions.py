"""Functions for data conversions."""

import numpy as np
import pandas as pd


def seconds_to_samples(time_value, fs_hz):
    """Convert a time value in seconds to the equivalent number of samples.

    Parameters
    ----------
    time_value : int, float, np.ndarray, pd.Series, pd.DataFrame, or iterable
        Time duration(s) in seconds.
    fs_hz : float
        Sampling frequency in Hz.

    Returns
    -------
    samples
        Corresponding number of samples as integer(s).
    """
    if isinstance(time_value, np.ndarray):
        return np.round(time_value * fs_hz).astype(int)
    elif isinstance(time_value, (int, float)):
        return int(np.round(time_value * fs_hz))
    elif isinstance(time_value, (pd.Series, pd.DataFrame)):
        return (time_value * fs_hz).round().astype(int)
    else:
        # For any other iterable (list, tuple, etc.)
        return type(time_value)(int(np.round(t * fs_hz)) for t in time_value)


def rename_axes_to_body(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rename all x, y, z columns to body-frame names (x -> is, y -> ml, z -> pa)
    and standardize column order: acc_is, acc_ml, acc_pa, gyr_is, gyr_ml, gyr_pa
    (only for columns present).
    """
    axis_map = {"x": "is", "y": "ml", "z": "pa"}

    rename_dict = {}
    for col in data.columns:
        if col.endswith(("_x", "_y", "_z")):
            prefix, axis = col.rsplit("_", 1)
            if axis in axis_map:
                rename_dict[col] = f"{prefix}_{axis_map[axis]}"

    df = data.rename(columns=rename_dict)

    # Desired order
    desired_order = ["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"]

    # Keep only those that exist in the DataFrame
    ordered_existing = [c for c in desired_order if c in df.columns]

    # Add the remaining columns in their original order
    remaining = [c for c in df.columns if c not in ordered_existing]

    return df[ordered_existing + remaining]
