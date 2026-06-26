"""
DMO threshold checks.

Assumptions:
- thresholds CSV at: files("multigait") / "pipeline/utils/dmo_thresholds.csv"
  has columns: dmo, min, max (first column is dmo name).
- Single min/max per DMO (no cohorts or sources).
- input_data is a DataFrame with DMO columns (e.g. 'cadence_spm',
  'walking_speed_mps', 'stride_length_m', ...). Only columns present in both
  thresholds and the input are checked; other columns in the result are NA.
"""

from importlib.resources import files
from typing import Optional
import numpy as np
import pandas as pd


def get_thresholds() -> pd.DataFrame:
    """
    Load simple DMO thresholds from the packaged CSV:
      files("multigait") / "pipeline/utils/dmo_thresholds.csv"

    Returns
    -------
    pd.DataFrame
        Indexed by DMO name with float columns ['min', 'max'].
    """
    path = files("multigait") / "pipeline/utils/dmo_thresholds.csv"
    with path.open() as fh:
        thr = pd.read_csv(fh, index_col=0)
    if not {"min", "max"}.issubset(thr.columns):
        raise ValueError("Thresholds CSV must contain 'min' and 'max' columns.")
    thr = thr[["min", "max"]].astype(float)
    thr.index.name = "dmo"
    return thr


def _max_allowable_stride_length(height_m: float) -> float:
    """
    Compute a physical upper bound for stride length from height (same model as original).

    Returns
    -------
    float
        Maximum allowable stride length (m).
    """
    leg_length = 0.53 * height_m
    froude_number = 1.0
    v_max = np.sqrt(froude_number * 9.81 * leg_length)
    max_vertical_displacement = 0.038 * v_max**2
    # Zijlstra style inverted-pendulum formula
    max_sl = (
        2
        * 2
        * np.sqrt(
            2 * leg_length * max_vertical_displacement - max_vertical_displacement**2
        )
    )
    return float(max_sl)


def apply_thresholds(
    input_data: pd.DataFrame,
    thresholds: Optional[pd.DataFrame] = None,
    *,
    height_m: Optional[float] = None,
) -> pd.DataFrame:
    """
    #TODO: update threshold values when lab dataset is finalised.
    Check which values in input_data are within the provided simple thresholds.
    Thresholds are derived from a laboratory dataset comprising multiple tests conducted in a multimorbid population.
    The test protocol included slow and fast walking, continuous prolonged walking, and simulated activities of daily living.

    Parameters
    - input_data: DataFrame with DMO columns (e.g. 'cadence_spm', 'stride_length_m', ...)
    - thresholds: DataFrame indexed by DMO name with ['min','max'].
                  If None, will load using get_simple_dmo_thresholds().
    - height_m: optional participant height (m). If provided, will increase the 'max'
                for 'stride_length_m' to be at least the physically computed maximum.

    Returns
    - DataFrame with the same index and columns as input_data. For each column that exists in
      thresholds, values are boolean: True if within [min,max] (inclusive), False if outside.
      Columns not present in thresholds are left with NA (pandas BooleanDtype with NA).
    """
    if thresholds is None:
        thresholds = get_thresholds()

    if not {"min", "max"}.issubset(thresholds.columns):
        raise ValueError("Thresholds must contain 'min' and 'max' columns.")

    thr = thresholds.copy()

    # Optionally adjust stride_length_m max by height-based physical limit
    if height_m is not None and "stride_length_m" in thr.index:
        computed_max_sl = _max_allowable_stride_length(height_m)
        thr.loc["stride_length_m", "max"] = max(
            thr.loc["stride_length_m", "max"], computed_max_sl
        )

    # DMOs to check: intersection of input columns and threshold index
    dmos_to_check = [c for c in input_data.columns if c in thr.index]

    # Prepare output frame with pandas BooleanDtype (supports NA)
    result = pd.DataFrame(
        index=input_data.index, columns=input_data.columns, dtype="boolean"
    )

    # Vectorised comparisons per column
    for dmo in dmos_to_check:
        lo = float(thr.loc[dmo, "min"])
        hi = float(thr.loc[dmo, "max"])
        mask = input_data[dmo].between(lo, hi, inclusive="both")
        # between returns boolean or NA for NaN inputs; cast to pandas BooleanDtype
        result[dmo] = mask.astype("boolean")

    return result
