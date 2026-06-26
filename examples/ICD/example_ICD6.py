from multigait.ICD import GuIC
from multigait.utils.data_loader import load_imu_data_wrist

"""
This is an example on how to use the Gu algo to detect initial contacts.
"""

imu_data = load_imu_data_wrist()

# only one bout of walking [662:2054]
imu_data = imu_data[662:2054]

# Create an instance of the GuIC class
ICs = GuIC(version="improved_wrist").detect(imu_data, sampling_rate_hz=100)

print(ICs.ic_list_)
