import pandas as pd


def cwb(df, max_break_seconds=3, sampling_rate=100):
    """
    Creating a Continuous Walking Bout (CWB) from micro walking bouts.
    Effectively merges walking bouts when the gap between them is shorter than 3 seconds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['start', 'end'] and sorted by start time.
    max_break_seconds : int
        Maximum allowed gap (in seconds) between bouts to merge them.
    sampling_rate : int
        Sampling rate in Hz, used to convert seconds to samples.

    Returns
    -------
    pd.DataFrame
        Merged bouts with columns ['start', 'end'], keeping index name 'gs_id'.
    """

    # Sort without reseting index
    df = df.sort_values("start")

    # Input empty → return empty with correct index name
    if df.empty:
        empty_df = pd.DataFrame(columns=["start", "end"])
        empty_df.index.name = df.index.name or "gs_id"
        return empty_df

    max_break = max_break_seconds * sampling_rate

    merged = []
    current_start = df.iloc[0]["start"]
    current_end = df.iloc[0]["end"]

    for i in range(1, len(df)):
        gap = df.iloc[i]["start"] - current_end

        if gap <= max_break:
            current_end = max(current_end, df.iloc[i]["end"])
        else:
            merged.append({"start": current_start, "end": current_end})
            current_start = df.iloc[i]["start"]
            current_end = df.iloc[i]["end"]

    # Append final bout
    merged.append({"start": current_start, "end": current_end})

    # Build output
    out = pd.DataFrame(merged)

    # Preserve or assign index name
    out.index.name = df.index.name or "gs_id"

    return out
