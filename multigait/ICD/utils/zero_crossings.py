import numpy as np
from typing import Literal


def detect_zero_crossings(
    data: np.ndarray, direction: Literal["pos_to_neg", "neg_to_pos", "both"] = "both"
) -> np.ndarray:
    """
    Detect zero crossings in a numerical array.

    The returned indices are floats if interpolation is applied, providing a more precise crossing location.

    Parameters
    ----------
    data : np.ndarray
        Input signal array.
    direction : str
        Specifies which zero crossings to detect:
        'pos_to_neg' for positive-to-negative transitions,
        'neg_to_pos' for negative-to-positive transitions,
        or 'both' to detect all crossings. Default is 'both'.

    Returns
    -------
    np.ndarray
        Array of indices where zero crossings occur.

    Raises
    ------
    ValueError
        If the `direction` argument is invalid.
    """
    signs = np.sign(data)
    # Treat zeros as positive to handle flat regions
    signs[signs == 0] = 1

    # Identify points where the sign changes
    zero_idx = np.where(np.diff(signs) != 0)[0]

    # Filter crossings according to the requested direction
    if direction == "pos_to_neg":
        zero_idx = zero_idx[data[zero_idx] >= 0]
    elif direction == "neg_to_pos":
        zero_idx = zero_idx[data[zero_idx] < 0]
    elif direction == "both":
        pass
    else:
        raise ValueError("direction must be 'pos_to_neg', 'neg_to_pos', or 'both'.")

    return zero_idx
