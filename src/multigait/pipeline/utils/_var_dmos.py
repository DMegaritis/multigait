"""
Helpers to compute within-walking-bout coefficient of variation (CV) and RMSSD
(Root Mean Square of Successive Differences) for per-stride parameters.

Improved validation and clearer error messages compared to the original version.

Intended usage:
    per_wb_metrics = within_wb_var(final_strides)
    per_wb_params = pd.concat([per_wb_params, per_wb_metrics], axis=1)

final_strides: DataFrame indexed by (wb_id, s_id) where level 0 is wb_id.
"""

from typing import Iterable, Optional, Sequence
import numpy as np
import pandas as pd

DEFAULT_COLS = [
    "stride_duration_s",
    "cadence_spm",
    "stride_length_m",
    "walking_speed_mps",
]


def _rmssd_series(series: pd.Series) -> float:
    """Compute RMSSD for a 1-D series (ignoring NA)."""
    dif = series.dropna().diff().dropna()
    if dif.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(dif.values**2)))


def within_wb_var(
    final_strides: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    ddof: int = 0,
) -> pd.DataFrame:
    """
    Compute per-walking-bout statistics for selected per-stride columns.

    Returned metrics:
      - <col>_cv   : coefficient of variation = std / mean (requires >= 2 samples)
      - <col>_rmssd: root mean square of successive differences (requires >= 2 samples)

    Parameters
    ----------
    final_strides
        DataFrame indexed by a MultiIndex (wb_id, s_id) or with wb_id as the first index level.
        Alternatively, a column named 'wb_id' is accepted and will be used for grouping (the original frame is not mutated).
    cols
        Iterable of column names to compute metrics for. If None, uses DEFAULT_COLS.
        Only columns present in final_strides are used.
    ddof
        Degrees of freedom for standard deviation (0 for population, 1 for sample).

    Returns
    -------
    DataFrame indexed by wb_id containing "<col>_cv" and "<col>_rmssd" columns.
    Groups with fewer than 2 non-NA samples will have NaN for both metrics.

    Raises
    ------
    TypeError
        If final_strides is not a pandas DataFrame or ddof is not an int.
    ValueError
        If the input does not contain a usable wb identifier (level 0 of index or 'wb_id' column),
        or if none of the requested cols exist in final_strides.
    """
    # Basic type checks
    if not isinstance(final_strides, pd.DataFrame):
        raise TypeError("final_strides must be a pandas DataFrame.")

    if not isinstance(ddof, int) or ddof < 0:
        raise TypeError("ddof must be a non-negative integer.")

    # Prepare columns
    if cols is None:
        cols = list(DEFAULT_COLS)
    else:
        # Convert to list to preserve order
        if isinstance(cols, Sequence) and not isinstance(cols, str):
            cols = list(cols)
        else:
            cols = list(cols)

    # Restrict to columns actually present
    cols_present = [c for c in cols if c in final_strides.columns]
    if not cols_present:
        # nothing to compute — return empty frame with unique wb ids
        # Determine wb ids if possible, otherwise return empty DataFrame
        if "wb_id" in final_strides.columns:
            wb_ids = pd.Index(final_strides["wb_id"].unique(), name="wb_id")
        else:
            # try to infer from index level 0; if that fails produce empty index
            try:
                wb_ids = final_strides.index.get_level_values(0).unique()
                # preserve a name if present
                wb_ids = pd.Index(wb_ids, name=final_strides.index.names[0] or "wb_id")
            except Exception:
                wb_ids = pd.Index([], name="wb_id")
        return pd.DataFrame(index=wb_ids)

    # Decide grouping: prefer explicit 'wb_id' column if present, otherwise group by level 0
    if "wb_id" in final_strides.columns:
        grouping = final_strides["wb_id"]
        wb_index_name = "wb_id"
        grouped = final_strides[cols_present].groupby(grouping)
    else:
        # Validate that level 0 exists and is meaningful
        try:
            # This will work for RangeIndex as well; we want to ensure the user expects level 0 as wb_id.
            level0_name = final_strides.index.names[0]
            grouped = final_strides[cols_present].groupby(level=0)
            wb_index_name = level0_name or "wb_id"
        except Exception:
            raise ValueError(
                "Could not determine wb identifier. Provide a DataFrame indexed by (wb_id, ...) "
                "or include a 'wb_id' column."
            )

    # Quick path: empty input
    if final_strides.shape[0] == 0:
        return pd.DataFrame(index=pd.Index([], name=wb_index_name))

    # Counts of non-NA values per group (used to enforce NaN for single-stride WBs)
    counts = grouped.count()

    # Means and stds
    mean = grouped.mean()
    std = grouped.std(ddof=ddof)

    # CV = std / mean; guard divide-by-zero and inf
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean
    cv = cv.replace([np.inf, -np.inf], np.nan)

    # enforce NaN for groups with fewer than 2 valid samples (one stride or zero)
    mask_too_few = counts < 2
    # mask_too_few has same shape/columns as cv; align and mask
    cv = cv.mask(mask_too_few)

    # Compute RMSSD for each group and column. The result of groupby.apply will have an index with wb_id.
    # We compute per-group by passing series to _rmssd_series via DataFrame.apply.
    rmssd = grouped.apply(lambda df: df.apply(_rmssd_series))
    # After groupby.apply the rmssd DataFrame may have a name in its index; ensure alignment with mean/cv indexes
    # If rmssd comes back as a Series (happens for single-column), convert to DataFrame
    if isinstance(rmssd, pd.Series):
        rmssd = (
            rmssd.unstack(level=-1)
            if isinstance(rmssd.index, pd.MultiIndex)
            else rmssd.to_frame()
        )

    # rmssd already yields NaN if not enough diffs, but ensure mask consistency
    # Reindex rmssd like mean (index and columns) to ensure consistent layout, then mask
    try:
        rmssd = rmssd.reindex(index=mean.index, columns=mean.columns)
    except Exception:
        # If reindexing fails for some reason, fall back to aligning via concat
        rmssd = pd.DataFrame(rmssd).reindex(index=mean.index, columns=mean.columns)

    rmssd = rmssd.mask(mask_too_few)

    # rename columns to indicate metric, preserving the order of cols_present
    cv = cv.rename(columns={c: f"{c}_cv" for c in cv.columns})
    rmssd = rmssd.rename(columns={c: f"{c}_rmssd" for c in rmssd.columns})

    # combine results side-by-side in order: for each original col, its cv then rmssd
    parts = []
    for c in cols_present:
        cv_col = f"{c}_cv"
        rmssd_col = f"{c}_rmssd"
        # If a metric column is missing (e.g., all-NA column), create it filled with NaN to preserve shape/order
        if cv_col not in cv.columns:
            cv_piece = pd.DataFrame(index=mean.index, columns=[cv_col], data=np.nan)
        else:
            cv_piece = cv[[cv_col]]
        if rmssd_col not in rmssd.columns:
            rmssd_piece = pd.DataFrame(
                index=mean.index, columns=[rmssd_col], data=np.nan
            )
        else:
            rmssd_piece = rmssd[[rmssd_col]]
        parts.append(cv_piece)
        parts.append(rmssd_piece)

    result = pd.concat(parts, axis=1)

    # ensure index name is set to wb id name for clarity
    if result.index.name is None:
        result.index.name = wb_index_name or "wb_id"

    return result
