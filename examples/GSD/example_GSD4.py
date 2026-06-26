from src.multigait import load_imu_data_lowback
from multigait.GSD import MacLeanGSD

"""
This is an example on how to use the MacLean algo to detect gait events.
"""

imu_data = load_imu_data_lowback()

# Creating instance of the class and calling the preprocess and detect methods
GSDs = MacLeanGSD(version="original_lowback").detect(imu_data)

print(GSDs.gs_list_)
