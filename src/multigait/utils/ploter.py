import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_interp(dfs, labels=None, max_points=5000, minute_tick_interval=5):
    """
    Plot the X-axis acceleration from multiple DataFrames, which may have columns
    'accel_x', 'acc_x', or 'acc_is', and either a 'time' column or datetime index.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        List of dataframes containing acceleration and time information.
    labels : list of str, optional
        Labels for each dataframe (defaults to ['df1', 'df2', ...]).
    max_points : int, optional
        Maximum points to plot per dataframe (default: 5000, for performance).
    minute_tick_interval : int, optional
        Major x-axis tick spacing in minutes (default: 5).
    """
    if labels is None:
        labels = [f"df{i + 1}" for i in range(len(dfs))]

    fig, ax = plt.subplots(figsize=(12, 6))

    for df, label in zip(dfs, labels):
        # Handle time column or datetime index
        if "time" in df.columns:
            t = df["time"]
        else:
            t = df.index

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(t):
            t = pd.to_datetime(t)

        # Determine acceleration column
        for col in ["accel_x", "acc_x", "acc_is"]:
            if col in df.columns:
                acc_col = col
                break
        else:
            raise ValueError(f"No acceleration column found in dataframe '{label}'")

        # Downsample for plotting
        n = len(df)
        if n == 0:
            continue
        step = max(1, n // max_points)
        ax.plot(
            t[::step],
            df[acc_col].iloc[::step],
            label=label,
            alpha=0.8,
            linewidth=0.7,
            rasterized=True,
        )

    # X-axis formatting
    locator = mdates.MinuteLocator(interval=minute_tick_interval)
    formatter = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(
        mdates.MinuteLocator(interval=max(1, minute_tick_interval // 2))
    )

    # Labels and grid
    ax.set_title("Acceleration (X-axis) from Multiple Datasets")
    ax.set_xlabel("Time")
    ax.set_ylabel("Acceleration (X)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.07, right=0.82, top=0.93, bottom=0.22)

    plt.show()
