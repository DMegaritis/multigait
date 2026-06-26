"""
Loads package CSV data.
"""

from importlib import resources
import pandas as pd


def load_imu_data_lowback() -> pd.DataFrame:
    with resources.open_text("examples.data", "data_lowback.csv") as f:
        return pd.read_csv(f)


def load_ICs_lowback() -> pd.DataFrame:
    with resources.open_text("examples.data", "ics_lowback.csv") as f:
        return pd.read_csv(f)


def load_imu_data_wrist() -> pd.DataFrame:
    with resources.open_text("examples.data", "data_wrist.csv") as f:
        return pd.read_csv(f)


def load_ICs_wrist() -> pd.DataFrame:
    with resources.open_text("examples.data", "ics_wrist.csv") as f:
        return pd.read_csv(f)


def load_imu_data_interpolation_lowback() -> pd.DataFrame:
    with resources.open_text("examples.data", "data_lowback_interpolation.csv") as f:
        return pd.read_csv(f)


def load_imu_data_interpolation_wrist() -> pd.DataFrame:
    with resources.open_text("examples.data", "data_wrist_interpolation.csv") as f:
        return pd.read_csv(f)
