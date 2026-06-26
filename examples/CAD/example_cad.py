from multigait.CAD.cad import Cadence, CadenceSimple
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

# Calling the cad simple
cad_from_ic = CadenceSimple()

cad_from_ic.calculate(
    imu_data,
    initial_contacts=reference_ic,
    sampling_rate_hz=100,
)
print(cad_from_ic.cadence_spm_)
