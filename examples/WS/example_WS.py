# %%
# Firstly we calculate cadence from ICs

from multigait.CAD.cad import Cadence
from multigait.utils.data_loader import load_imu_data_wrist, load_ICs_wrist

imu_data = load_imu_data_wrist()
# only one bout of walking for the lowback data: 962:1427; for the wrist data 662:2054
imu_data = imu_data[662:2054]

reference_ic = load_ICs_wrist()

# calling the cad
cad_from_ic = Cadence()

cad_from_ic.calculate(
    imu_data,
    initial_contacts=reference_ic,
    sampling_rate_hz=100,
)
print(cad_from_ic.cadence_per_sec_)


# %%
# Then we calculate stride length
from multigait.SL.SL1 import WeinbergSL
from multigait.utils.data_loader import load_imu_data_wrist, load_ICs_wrist


# calling the stride length algorithm
sl = WeinbergSL(version="wrist").calculate(
    data=imu_data, initial_contacts=reference_ic, sampling_rate_hz=100
)

print(sl.stride_length_per_sec_)

# %%
# Now we caclulate the WS
from multigait.WS.walking_speed import Ws

ws = Ws().calculate(
    cadence_per_sec=cad_from_ic.cadence_per_sec_,
    stride_length_per_sec=sl.stride_length_per_sec_,
)

print(ws.walking_speed_per_sec_)
