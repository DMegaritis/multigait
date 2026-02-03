import numpy as np

def peakfind(x, ws):
    """
    Find local maxima in a signal using a sliding window approach.

    Parameters
    ----------
    x : array-like
        Input signal.
    ws : int
        Window size (must be odd, at least 3). If larger than the signal length,
        it will be reduced automatically.

    Returns
    -------
    maxout : ndarray
        Indices and values of local maxima as a 2D array of shape (n_peaks, 2).
        If no peaks are found, returns an empty array with shape (0, 2).

    Notes
    -----
    - Ensures safe operation even when the window size exceeds the signal length.
    - Returns integer indices for peaks.
    """

    x = np.asarray(x).flatten()
    n = len(x)

    # Adjust window if larger than signal
    ws = min(ws, n)
    if ws % 2 == 0:
        ws += 1
    ws = max(ws, 3)

    # Padding to handle boundary effects
    npad = ws // 2
    x_padded = np.pad(x, (npad, npad), mode='constant', constant_values=np.nan)

    locmax = []
    for i in range(npad, len(x_padded) - npad):
        window = x_padded[i - npad:i + npad + 1]
        center_value = window[npad]
        if center_value > np.nanmax(np.delete(window, npad)):
            locmax.append(i - npad)

    # Remove artificial peaks at the boundaries
    locmax = [idx for idx in locmax if 0 < idx < n]

    # Convert to integer array
    locmax = np.array(locmax, dtype=int)

    # Return empty array safely if no peaks found
    if locmax.size == 0:
        return np.empty((0, 2))

    # Return indices and values
    maxout = np.column_stack((locmax, x[locmax]))
    return maxout
