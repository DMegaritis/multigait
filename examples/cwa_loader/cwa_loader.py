# Need to manually install cwa_reader_rs as it is not included in pyproject.toml

from multigait.data_loader import CWADataset

# Loading all data
dataset = CWADataset("/Users/klch3/Documents/multimobility/free_living")
all_data = dataset.load()

# Access a single datum: wrist data for MM_001
df_wrist = dataset.get_sensor_data("MM_001", "Wrist")
print(df_wrist)

# Itterating over all data
for pid, sensors in all_data.items():
    print(f"Participant: {pid}")
    for sensor_name, sensor_info in sensors.items():
        df = sensor_info["data"]
        print(f"  Sensor: {sensor_name}, rows: {len(df)}")
        print(df)
