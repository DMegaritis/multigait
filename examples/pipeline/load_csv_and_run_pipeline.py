"""
Example: run the MultiGait pipeline on a raw wrist accelerometer CSV.

Input: a CSV with no header and one row per sample, with the columns
timestamp, acc_is, acc_ml, acc_pa. Acceleration must be in m/s^2 and the
sampling rate 100 Hz. The timestamp column is parsed as a datetime index.

The pipeline is run once per calendar day and the results are saved to CSV.

Participant anthropometrics (height, leg length, etc.) are not needed to run
the pipeline. They can be passed through participant_metadata, where some
algorithms can use them.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from multigait.pipeline import MultiGaitPipelineMultimorbidityImpaired

CSV_PATH = "data.csv"
OUT_DIR = Path("results")
PARTICIPANT_ID = "participant_01"
SAMPLING_RATE_HZ = 100


# The datapoint object that MultiGait expects
@dataclass
class MyGaitDatapoint:
    data: dict
    data_ss: pd.DataFrame
    participant_metadata: dict
    recording_metadata: dict
    participant_id: str
    group_label: dict
    sampling_rate_hz: float


def load_datapoint(
    csv_path: str,
    participant_id: str,
    sampling_rate_hz: float,
    test: str = "free_living",
    trial: str = "1",
) -> MyGaitDatapoint:
    # Datetime index and exactly these three axis names
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["time", "acc_is", "acc_ml", "acc_pa"],
        index_col="time",
        parse_dates=True,
    )

    return MyGaitDatapoint(
        data={"Wrist": df},
        data_ss=df,
        participant_metadata={},
        recording_metadata={
            "start_date_time_iso": df.index[0].isoformat(),
            "sampling_rate_hz": sampling_rate_hz,
            "recording_identifier": (test, trial),
        },
        participant_id=participant_id,
        group_label={"participant_id": participant_id, "test": test, "trial": trial},
        sampling_rate_hz=sampling_rate_hz,
    )


dp = load_datapoint(CSV_PATH, PARTICIPANT_ID, SAMPLING_RATE_HZ)

# One pipeline run per calendar day
per_wb, aggregated, failed = {}, {}, {}

for day, day_df in dp.data_ss.groupby(dp.data_ss.index.normalize()):
    day_dp = replace(
        dp,
        data={"Wrist": day_df},
        data_ss=day_df,
        recording_metadata={
            **dp.recording_metadata,
            "start_date_time_iso": day_df.index[0].isoformat(),
        },
    )
    try:
        pipeline = MultiGaitPipelineMultimorbidityImpaired()
        pipeline.safe_run(day_dp)
    except Exception as e:
        failed[day.date()] = repr(e)
        continue

    per_wb[day.date()] = pipeline.per_wb_parameters_  # one row per walking bout
    aggregated[day.date()] = pipeline.aggregated_parameters_  # daily DMOs

# Save results
OUT_DIR.mkdir(parents=True, exist_ok=True)

daily = pd.concat(aggregated, names=["date", "_"]).droplevel("_")
daily.to_csv(OUT_DIR / "daily_dmos.csv")
pd.concat(per_wb, names=["date"]).to_csv(OUT_DIR / "per_wb_all_days.csv")

if failed:
    pd.Series(failed, name="error").to_csv(OUT_DIR / "failed_days.csv")

print(f"Saved {len(daily)} days to {OUT_DIR}; {len(failed)} days failed.")
