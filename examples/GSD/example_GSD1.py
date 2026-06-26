from multigait.utils.data_loader import load_imu_data_wrist
from multigait.GSD.GSD1 import IonescuGSD

"""
This is an example on how to use the Ionescu algo to detect gait events from wrist data.
"""

imu_data = load_imu_data_wrist()

# Calling the class with the preprocess and detect at once
GSDs = IonescuGSD(version="wrist").detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)
