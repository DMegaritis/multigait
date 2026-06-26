import pandas as pd
import numpy as np


def compute_alpha_mle(
    df: pd.DataFrame, duration_col: str = "duration_s"
) -> pd.DataFrame:
    """
    Compute alpha (power-law exponent) from walking bout durations using MLE,
    add it as a column to the existing DataFrame, and return the DataFrame.

    Robust to empty input or non-positive/NaN durations:
      - If df is empty -> returns a copy with 'alpha' column (NaN) added.
      - If no valid positive durations -> returns copy with 'alpha' column (NaN).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing walking bout durations.
    duration_col : str
        Column name with bout durations (default: 'duration_s').

    Returns
    -------
    pd.DataFrame
        Original DataFrame (copied) with a new column 'alpha' added.
        If alpha cannot be computed, values will be np.nan.
    """
    # Defensive copy early to avoid mutating user's DataFrame
    df_out = df.copy()

    # If empty DataFrame -> add alpha column (NaN) and return
    if df_out.empty:
        df_out["alpha"] = np.nan
        return df_out

    # Check duration column exists
    if duration_col not in df_out.columns:
        raise KeyError(f"duration column '{duration_col}' not found in dataframe")

    # Extract durations and filter to positive, finite values
    x_all = pd.to_numeric(df_out[duration_col], errors="coerce")
    x = x_all.dropna()
    x = x[x > 0].values  # keep strictly positive values only

    # If no valid durations remain -> add alpha column (NaN) and return
    if len(x) == 0:
        df_out["alpha"] = np.nan
        return df_out

    # Compute xmin and MLE for alpha, but guard against degenerate cases
    xmin = np.min(x)
    n = len(x)

    # Denominator: sum(log(x / xmin)). If zero (e.g. all x == xmin) -> undefined
    denom = np.sum(np.log(x / xmin))

    if denom == 0:
        alpha = np.nan
    else:
        alpha = 1.0 + n / denom

    # Add alpha (same value for all rows) and return
    df_out["alpha"] = alpha

    return df_out
