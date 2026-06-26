"""Helper functions for common array manipulations."""

from numpy.lib.stride_tricks import sliding_window_view as np_stride_window
from typing import Optional
import numba
import numpy as np
from numba import njit


def create_sliding_windows(
    array: np.ndarray, window_size_samples: int, overlap_samples: int
) -> np.ndarray:
    """Generate overlapping windows along the first axis of the array.

    This produces a view of the array, not a copy. Modifying the output will affect the original array.

    Excess samples that do not fit into the last window are discarded.

    Parameters
    ----------
    array : np.ndarray
        The input data. Only the first axis is windowed.
    window_size_samples : int
        Number of samples per window.
    overlap_samples : int
        Number of overlapping samples between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of sliding windows with shape (num_windows, window_size_samples, ...).

    Notes
    -----
    For an input shape of (T, D1, D2, ...), output will have shape (num_windows, window_size_samples, D1, D2, ...),
    preserving the original shape of each time slice.
    """
    if overlap_samples >= window_size_samples:
        raise ValueError("overlap_samples must be smaller than window_size_samples")

    # Calculate the stride (step) between windows
    stride = window_size_samples - overlap_samples

    # Create the sliding window view and apply the stride
    windowed_view = np_stride_window(
        array, window_shape=(window_size_samples,), axis=0
    )[::stride]

    if array.ndim > 1:
        # Move the window dimension after the first axis to maintain consistent shape
        windowed_view = np.moveaxis(windowed_view, -1, 1)

    return windowed_view


def merge_interval(input_intervals: np.ndarray, gap_size: int = 0) -> np.ndarray:
    """Combine overlapping or closely spaced intervals into single intervals.

    Intervals with overlap or distance <= gap_size are merged.

    Parameters
    ----------
    input_intervals : np.ndarray of shape (n, 2)
        Array of intervals, each row is [start, end].
    gap_size : int
        Maximum allowed gap between intervals to merge them. Default is 0.

    Returns
    -------
    np.ndarray of shape (m, 2)
        Array of merged intervals.

    Examples
    --------
    >>> arr = np.array([[1, 3], [2, 4], [6, 8], [5, 7], [10, 12], [11, 15], [18, 20]])
    >>> merge_intervals(arr)
    array([[ 1,  4],
           [ 5,  8],
           [10, 15],
           [18, 20]])

    >>> merge_intervals(arr, 2)
    array([[ 1, 15],
           [18, 20]])
    """
    if input_intervals.shape[0] == 0:
        return input_intervals

    # Sort rows by start values using stable sort
    sorted_intervals = np.sort(input_intervals, axis=0, kind="stable")
    return np.array(_merge_intervals_numba(sorted_intervals, gap_size))


@njit
def _merge_intervals_numba(
    sorted_intervals: np.ndarray, gap_size: int
) -> numba.typed.List:
    """Numba-accelerated merge for sorted intervals."""
    merged = numba.typed.List()
    merged.append(sorted_intervals[0])

    for i in range(1, len(sorted_intervals)):
        current = sorted_intervals[i]
        last = merged[-1]

        # Merge if current interval starts before last ends + gap
        if last[0] <= current[0] <= last[1] + gap_size <= current[1] + gap_size:
            last[1] = current[1]
        else:
            merged.append(current)

    return merged


def bool_array_to_start_end(bool_array: np.ndarray) -> np.ndarray:
    """Convert a boolean array into an array of start-end intervals.

    Each contiguous True region is converted to [start, end), where `end` is exclusive.

    Parameters
    ----------
    bool_array : np.ndarray, shape (n,)
        Array with boolean values (True/False or 1/0).

    Returns
    -------
    np.ndarray of shape (m, 2)
        Array of start and end indices for each True region.

    Examples
    --------
    >>> arr = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
    >>> bool_array_to_start_end_array(arr)
    array([[2, 4],
           [6, 9]])
    """
    if not isinstance(bool_array, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean")

    if len(bool_array) == 0:
        return np.array([])

    slices = np.ma.flatnotmasked_contiguous(np.ma.masked_equal(bool_array, 0))
    return np.array([[s.start, s.stop] for s in slices])


def start_end_array_to_bool(
    start_end_array: np.ndarray, pad_to_length: Optional[int] = None
) -> np.ndarray:
    """Convert a start-end interval array back to a boolean array.

    The intervals are inclusive at start and exclusive at end: [start, end).

    Parameters
    ----------
    start_end_array : np.ndarray of shape (n, 2)
        Array of start and end indices.
    pad_to_length : int, optional
        Desired length of the output boolean array. If None, uses maximum end index.

    Returns
    -------
    np.ndarray of shape (pad_to_length,)
        Boolean array representing all intervals.

    Examples
    --------
    >>> arr = np.array([[3, 5], [7, 8]])
    >>> start_end_array_to_bool_array(arr, pad_to_length=12)
    array([False, False, False,  True,  True, False, False,  True, False,
           False, False, False])
    """
    start_end_array = np.atleast_2d(start_end_array)

    if pad_to_length is None:
        n_elements = start_end_array.max() if start_end_array.size > 0 else 0
    else:
        if pad_to_length < 0:
            raise ValueError("pad_to_length must be non-negative")
        n_elements = pad_to_length

    bool_array = np.zeros(n_elements, dtype=bool)
    for start, end in start_end_array:
        bool_array[start:end] = True

    return bool_array
