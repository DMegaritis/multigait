from src.multigait import load_imu_data_wrist
from src.multigait import HickeyGSD

"""
This is an example on how to use the Hickey algo to detect gait events.
"""

imu_data = load_imu_data_wrist()

# Calling the class with the preprocess and detect at once
GSDs = HickeyGSD(version="wrist").detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)
