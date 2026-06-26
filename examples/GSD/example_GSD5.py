from src.multigait import load_imu_data_wrist
from multigait.GSD import KerenGSD

"""
This is an example on how to use the Keren algo to detect gait events.
"""

imu_data = load_imu_data_wrist()

# Creating instance of the class and calling the preprocess and detect methods
GSDs = KerenGSD().detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)
