import pandas as pd
import numpy as np
import warnings
from scipy.interpolate import PchipInterpolator
from typing import Iterable


class Interpolation:
    """
    Align multiple time-series datasets to a uniform time grid using PCHIP interpolation.

    This class allows for multiple input DataFrames, each of which must have a 'time' column.
    The 'time' column should ideally have nanosecond precision (up to 9 fractional digits).
    Timestamps with fewer than 9 digits will be padded with zeros, and extra digits beyond 9
    will be truncated automatically.

    The interpolation process uses a Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
    for each column (excluding 'time'). After computing the PCHIP interpolation, the original
    data values are injected at the nearest grid points where timestamps match, preserving
    the original samples as much as possible.

    Features:
    - Any number or name of data columns is allowed; only the 'time' column is used for interpolation.
    - Two modes of handling alignment:
        1. `overlap_windows=True`: Crops all DataFrames to the overlapping time window across datasets.
        2. Default (`overlap_windows=False`): Fills zeros outside the range of each original dataset to
           produce DataFrames of equal length.
    - Automatically removes non-increasing timestamps within each DataFrame.
    - Returns interpolated DataFrames with the same column order across all inputs.
    - As a result, the output DataFrames have the same shape and can be concatenated/compared easily.

    Attributes:
        None (class is stateless; all operations are performed in the `interpolate` method)

    Usage:
        interp = Interpolation()
        aligned_dfs = interp.interpolate([df1, df2, df3], sampling_rate_hz=100, overlap_windows=True)
        # Returns a tuple of interpolated DataFrames: (df1_aligned, df2_aligned, df3_aligned)

    Notes:
    - Despite injecting the original values at the nearest grid point, the PCHIP interpolation might introduce interpolated values outside the original range,
     effectively smothing some peaks but signal morphology is not affected. This is mitigated by injecting original values at the nearest grid point but there might still be some distortion.
    """

    def __init__(self):
        pass

    def interpolate(
        self,
        dfs: Iterable[pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame],
        *,
        sampling_rate_hz: float = 100,
        overlap_windows: bool = False,
    ) -> tuple[pd.DataFrame, ...]:
        # Backwards-compatible: accept (df1, df2) as two positional args too
        if not isinstance(dfs, (list, tuple)):
            raise TypeError("dfs must be a list or tuple of pandas.DataFrame")
        if len(dfs) < 1:
            raise ValueError("Provide at least one DataFrame")

        # ** Clean input dataframes individually
        clean_dfs = []
        starts = []
        ends = []
        for df in dfs:
            df = df.copy()
            if "time" not in df.columns:
                raise ValueError("Each DataFrame must have a 'time' column")
            if not np.issubdtype(df["time"].dtype, np.datetime64):
                df["time"] = pd.to_datetime(df["time"])
            # remove non-increasing timestamps (keep strictly increasing)
            df = df[df["time"].diff().fillna(pd.Timedelta(seconds=1)) > pd.Timedelta(0)]
            df = df.reset_index(drop=True)
            if df.shape[0] == 0:
                warnings.warn(
                    "One of the input DataFrames became empty after removing non-increasing timestamps.",
                    UserWarning,
                )
            clean_dfs.append(df)
            if df.shape[0] > 0:
                starts.append(df["time"].iloc[0])
                ends.append(df["time"].iloc[-1])
            else:
                # placeholder times so global bounds ignore empty dfs
                starts.append(pd.Timestamp.max)
                ends.append(pd.Timestamp.min)

        # Use cleaned dfs from now on
        dfs = clean_dfs

        # ** Check overlap across all non-empty dfs
        non_empty_indices = [i for i, df in enumerate(dfs) if df.shape[0] > 0]
        overlap = True
        if len(non_empty_indices) == 0:
            # nothing to interpolate: return empty frames
            return tuple(pd.DataFrame(columns=[]) for _ in dfs)

        latest_start = max(starts[i] for i in non_empty_indices)
        earliest_end = min(ends[i] for i in non_empty_indices)
        print(latest_start, earliest_end)
        if latest_start > earliest_end:
            overlap = False
            warnings.warn(
                "No temporal overlap between datasets:\n"
                + "\n".join(
                    f"  df{i} range = [{(dfs[i]['time'].iloc[0] if dfs[i].shape[0] > 0 else 'empty')}, "
                    f"{(dfs[i]['time'].iloc[-1] if dfs[i].shape[0] > 0 else 'empty')}]"
                    for i in range(len(dfs))
                ),
                UserWarning,
            )

        # ** Build union of columns (excluding 'time') so outputs share same columns
        all_cols = []
        for df in dfs:
            cols = [c for c in df.columns if c != "time"]
            for c in cols:
                if c not in all_cols:
                    all_cols.append(c)
        cols = all_cols  # final ordered list of columns
        n_cols = len(cols)

        # ** Build target time grid across min start and max end (ignoring empty dfs)
        t_start = min(starts[i] for i in non_empty_indices)
        t_end = max(ends[i] for i in non_empty_indices)
        duration_s = (t_end - t_start).total_seconds()
        n_samples = max(1, int(duration_s * sampling_rate_hz))

        if n_samples == 1:
            t_target_ns = np.array([t_start.value], dtype=np.int64)
        else:
            t_target_float = np.linspace(
                float(t_start.value), float(t_end.value), n_samples
            )
            t_target_ns = np.rint(t_target_float).astype(np.int64)
        t_target = pd.to_datetime(t_target_ns)

        # ** Preallocate arrays for interpolated values and masks (one per input df)
        arrays_interp = [
            np.zeros((n_samples, n_cols), dtype=float) for _ in range(len(dfs))
        ]
        masks = [np.zeros(n_samples, dtype=bool) for _ in range(len(dfs))]

        # ** For each df, do PCHIP interpolation for the columns that exist in that df.
        for i, df in enumerate(dfs):
            if df.shape[0] == 0:
                # leave as zeros and mask stays False
                continue

            x_ns = df["time"].values.astype("int64")

            # mask True for points inside original data range
            masks[i] = (t_target_ns >= x_ns.min()) & (t_target_ns <= x_ns.max())

            # For columns not present in df, we keep zeros (consistent with outside-range fill)
            for j, col in enumerate(cols):
                if col not in df.columns:
                    # column absent in this df: leave zeros
                    continue
                y = df[col].values.astype(float)
                # only interpolate if we have >=2 points; else rely on injection of originals
                if df.shape[0] >= 2:
                    try:
                        interp_fn = PchipInterpolator(
                            x_ns.astype(float), y, extrapolate=False
                        )
                        y_interp = interp_fn(t_target_ns.astype(float))
                    except Exception:
                        # fallback: set zeros and rely on injection
                        y_interp = np.zeros(n_samples, dtype=float)
                    # fill outside original data range with 0 (explicit)
                    y_interp[t_target_ns < x_ns.min()] = 0.0
                    y_interp[t_target_ns > x_ns.max()] = 0.0
                    arrays_interp[i][:, j] = y_interp
                else:
                    # single-sample df: leave zeros for now; injection below will add the original value
                    arrays_interp[i][:, j] = 0.0

        # ** Inject original sample values at nearest grid index (average if multiple map to same index)
        def inject_originals(arr_interp, df):
            if df.shape[0] == 0:
                return arr_interp
            x_ns = df["time"].values.astype("int64")
            if n_samples == 1:
                idx = np.zeros_like(x_ns, dtype=np.int64)
            else:
                step = float(t_target_ns[1] - t_target_ns[0])
                rel = (x_ns - t_target_ns[0]).astype(np.float64) / step
                idx = np.rint(rel).astype(np.int64)
                idx = np.clip(idx, 0, n_samples - 1)

            # For each column present in df, average values per index and place them in arr_interp
            for j, col in enumerate(cols):
                if col not in df.columns:
                    continue
                sums = np.zeros(n_samples, dtype=float)
                counts = np.zeros(n_samples, dtype=int)
                vals = df[col].values.astype(float)
                for k, idk in enumerate(idx):
                    sums[idk] += vals[k]
                    counts[idk] += 1
                mask_idx = counts > 0
                if np.any(mask_idx):
                    avg = np.zeros_like(sums)
                    avg[mask_idx] = sums[mask_idx] / counts[mask_idx]
                    arr_interp[mask_idx, j] = avg[mask_idx]
            return arr_interp

        for i, df in enumerate(dfs):
            arrays_interp[i] = inject_originals(arrays_interp[i], df)

        # ** Build DataFrames with mask column for cropping logic
        dfs_interp = []
        for i in range(len(dfs)):
            df_interp = pd.DataFrame(arrays_interp[i], columns=cols, index=t_target)
            df_interp["valid"] = masks[i]
            dfs_interp.append(df_interp)

        # ** Crop to overlapping window across all dfs' valid masks if requested
        if overlap_windows:
            if overlap:
                mask_overlap = np.ones(n_samples, dtype=bool)
                for m in masks:
                    mask_overlap &= m
                # if mask_overlap all False -> result will be empty
                dfs_interp = [df.iloc[mask_overlap] for df in dfs_interp]
            else:
                warnings.warn(
                    "Cannot crop to overlapping window because the datasets have no temporal overlap.",
                    UserWarning,
                )

        # ** Drop the mask column before returning
        for i in range(len(dfs_interp)):
            if "valid" in dfs_interp[i].columns:
                dfs_interp[i] = dfs_interp[i].drop(columns="valid")

        return tuple(dfs_interp)
