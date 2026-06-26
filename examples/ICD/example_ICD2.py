from src.multigait.ICD.ICD2 import McCamleyIC
from src.multigait import load_imu_data_lowback

"""
This is an example on how to use the McCamley algo to detect initial contacts.
"""

imu_data = load_imu_data_lowback()

# only one bout of walking for the lowback data: 962:1427; for the wrist data 662:2054
imu_data = imu_data[962:1427]

# Create an instance of the McCamley class
ICs = McCamleyIC(version="improved_lowback").detect(imu_data, sampling_rate_hz=100)

print(ICs.ic_list_)
