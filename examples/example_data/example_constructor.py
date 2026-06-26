import os
import pandas as pd
import ast


class MyGaitDatapoint:
    def __init__(
        self,
        data,
        data_ss,
        participant_metadata,
        recording_metadata,
        participant_id,
        group_label,
        sampling_rate_hz,
    ):
        self.data = data
        self.data_ss = data_ss
        self.participant_metadata = participant_metadata
        self.recording_metadata = recording_metadata
        self.participant_id = participant_id
        self.group_label = group_label
        self.sampling_rate_hz = sampling_rate_hz


def construct_datapoint_from_files(
    wrist_csv_filename="data_wrist.csv", metadata_txt_filename="metadata.txt"
):
    # always use the script directory, relative to current file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrist_csv_path = os.path.join(script_dir, wrist_csv_filename)
    metadata_txt_path = os.path.join(script_dir, metadata_txt_filename)

    # Load the wrist CSV
    wrist_df = pd.read_csv(wrist_csv_path, index_col=0, parse_dates=True)

    # Parse the metadata txt expecting a python dictionary as text
    with open(metadata_txt_path, "r") as f:
        meta_dict = ast.literal_eval(f.read())

    # Extract participant_id, remove from main metadata if needed
    participant_id = meta_dict.pop("participant_id")

    # Split metadata fields
    recording_fields = [
        "start_date_time_iso",
        "sampling_rate_hz",
        "recording_identifier",
    ]
    participant_fields = [
        "arm_length_cm",
        "foot_length_cm",
        "height_m",
        "leg_length_cm",
        "sensor_height_m",
        "sensor_locations",
        "shoe_length_cm",
    ]

    recording_metadata = {k: meta_dict[k] for k in recording_fields if k in meta_dict}
    participant_metadata = {
        k: meta_dict[k] for k in participant_fields if k in meta_dict
    }

    # Extract sampling_rate_hz
    sampling_rate_hz = recording_metadata.get("sampling_rate_hz", None)

    # Create group_label dict
    rec_id = recording_metadata.get("recording_identifier", ("", ""))
    test = rec_id[0] if isinstance(rec_id, (list, tuple)) and len(rec_id) > 0 else None
    trial = rec_id[1] if isinstance(rec_id, (list, tuple)) and len(rec_id) > 1 else None
    group_label = {"participant_id": participant_id, "test": test, "trial": trial}

    # Construct 'data' dict (only 'Wrist')
    data_dict = {"Wrist": wrist_df}

    return MyGaitDatapoint(
        data=data_dict,
        data_ss=wrist_df,
        participant_metadata=participant_metadata,
        recording_metadata=recording_metadata,
        participant_id=participant_id,
        group_label=group_label,
        sampling_rate_hz=sampling_rate_hz,
    )
