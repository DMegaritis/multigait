from multigait.ICD.ICD5 import DucharmeIC
from multigait.utils.data_loader import load_imu_data_lowback

"""
This is an example on how to use the Ducharme algo to detect initial contacts.
"""

imu_data = load_imu_data_lowback()

# only one bout of walking for the lowback data: 962:1427; for the wrist data 662:2054
imu_data = imu_data[962:1427]

# Create an instance of the EncarnaIC class
ICs = DucharmeIC(version="original_lowback").detect(imu_data, sampling_rate_hz=100)

print(ICs.ic_list_)
