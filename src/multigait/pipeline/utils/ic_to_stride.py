import pandas as pd


def _to_stride_list_no_lr(ic_list: pd.DataFrame) -> pd.DataFrame:
    """Make start/end stride rows from an ic-only DataFrame.

    Expects a DataFrame with column "ic". Returns rows where start=ic and
    end=next ic (last row is dropped).
    """
    return (
        ic_list[["ic"]]
        .rename(columns={"ic": "start"})
        .assign(end=lambda df_: df_["start"].shift(-1))
        .dropna()
        .astype({"start": "int64", "end": "int64"})
    )


def _unify_stride_list_no_lr(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtype/order and set index name to 's_id' (or last level name -> 's_id')."""
    df = df.astype({"start": "int64", "end": "int64"})[["start", "end"]]
    if isinstance(df.index, pd.MultiIndex):
        # rename the last level of the MultiIndex to 's_id'
        df.index = df.index.rename("s_id", level=-1)
    else:
        df.index.name = "s_id"
    return df


def strides_list_from_ic_list_no_lrc(ic_list: pd.DataFrame) -> pd.DataFrame:
    """Convert an initial contact list (no laterality) to a stride list.

    Each stride goes from one IC to the next IC (consecutive ICs).
    The function preserves the input index structure (so it works when applied
    via groupby on gs_id).
    """
    # If empty: return empty frame with compatible index/shape
    if ic_list.empty:
        return pd.DataFrame(columns=["start", "end"], index=ic_list.index).pipe(
            _unify_stride_list_no_lr
        )

    # Ensure sorted by ic and compute consecutive pairs
    return (
        ic_list.sort_values("ic")
        .pipe(_to_stride_list_no_lr)
        .pipe(_unify_stride_list_no_lr)
    )
