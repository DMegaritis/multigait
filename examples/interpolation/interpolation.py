"""In this example we use very short data to show the interpolation."""

from src.multigait import Interpolation
from src.multigait import plot_interp
from src.multigait import (
    load_imu_data_interpolation_lowback,
    load_imu_data_interpolation_wrist,
)

lowback_data = load_imu_data_interpolation_lowback()
wrist_data = load_imu_data_interpolation_wrist()

# Ploting:
plot_interp([lowback_data, wrist_data], labels=["lowback_data", "wrist_data"])

# Interpolation
dfs = [lowback_data, wrist_data]
interpolated = Interpolation().interpolate(dfs, overlap_windows=True)

lowback_data_interpolated, wrist_data_interpolated = interpolated

plot_interp(
    [lowback_data_interpolated, wrist_data_interpolated],
    labels=["lowback_data_interpolated", "wrist_data_interpolated"],
)
