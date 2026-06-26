from src.multigait.SL.SL2 import KimSL
from src.multigait import load_imu_data_wrist, load_ICs_wrist

"""
This is an example on how to use the intensity based Kim stride length algorithm.
"""

imu_data = load_imu_data_wrist()
# only one bout of walking for the lowback data: 962:1427; for the wrist data 662:2054
imu_data = imu_data[662:2054]

reference_ic = load_ICs_wrist()

# calling the stride length algorithm
sl = KimSL(version="wrist").calculate(
    data=imu_data, initial_contacts=reference_ic, sampling_rate_hz=100
)

print(sl.stride_length_per_sec_)
print(sl.average_stride_length_)
