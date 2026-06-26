from src.multigait import load_imu_data_lowback
from multigait.GSD import KheirkhahanGSD

"""
This is an example on how to use the Kheirkhahan algo to detect gait events.
"""

imu_data = load_imu_data_lowback()

# Creating instance of the class and calling the preprocess and detect methods
GSDs = KheirkhahanGSD(version="original_lowback").detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)
