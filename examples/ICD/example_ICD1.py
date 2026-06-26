from multigait.ICD.ICD1 import MicoAmigoIC
from multigait.utils.data_loader import load_imu_data_wrist

"""
This is an example on how to use the MicoAmigo algo to detect initial contacts.
"""

imu_data = load_imu_data_wrist()

# only one bout of walking for the lowback data: 962:1427; for the wrist data 662:2054
imu_data = imu_data[662:2054]

# Create an instance of the MicoAmigoIC class
ICs = MicoAmigoIC(version="wrist").detect(imu_data, sampling_rate_hz=100)
print(ICs.ic_list_)
