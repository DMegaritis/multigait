import pandas as pd


def check_gait_datapoint_completeness(datapoint, verbose=True) -> bool:
    """
    Check if a gait datapoint has all required attributes and required fields,
    reflecting all characteristics in gait_data_processor.py.

    Parameters
    ----------
    datapoint : object
        The datapoint object to check.
    verbose : bool
        If True, prints a list of missing or incomplete attributes.

    Returns
    -------
    is_complete : bool
        True if all required fields are present and valid, False otherwise.
    """
    missing = []

    # Top-level attributes required by MyGaitDatapoint
    expected_attrs = [
        "data_ss",
        "data",
        "participant_metadata",
        "recording_metadata",
        "participant_id",
        "group_label",
        "sampling_rate_hz",
    ]
    for attr in expected_attrs:
        if not hasattr(datapoint, attr):
            missing.append(attr)
        elif getattr(datapoint, attr) is None:
            missing.append(f"{attr} (is None)")

    # .data: Should be a dict with 'LowerBack' and 'Wrist' keys and pd.DataFrame values
    if hasattr(datapoint, "data") and getattr(datapoint, "data") is not None:
        data_dict = getattr(datapoint, "data")
        if not isinstance(data_dict, dict):
            missing.append("data (not a dict)")
        else:
            for key in ["LowerBack", "Wrist"]:
                if key not in data_dict:
                    missing.append(f"data['{key}'] (missing)")
                else:
                    df = data_dict[key]
                    if not isinstance(df, pd.DataFrame):
                        missing.append(f"data['{key}'] (not a DataFrame)")
                    elif df.empty:
                        missing.append(f"data['{key}'] (DataFrame is empty)")
    else:
        missing.append("data (attribute missing or None)")

    # participant_metadata fields
    participant_fields = [
        "arm_length_cm",
        "foot_length_cm",
        "height_m",
        "leg_length_cm",
        "sensor_height_m",
        "sensor_locations",
        "shoe_length_cm",
    ]
    if (
        hasattr(datapoint, "participant_metadata")
        and getattr(datapoint, "participant_metadata") is not None
    ):
        meta = getattr(datapoint, "participant_metadata")
        for field in participant_fields:
            if field not in meta:
                missing.append(f"participant_metadata.{field} (missing)")
            elif meta[field] is None:
                missing.append(f"participant_metadata.{field} (is None)")
    else:
        for field in participant_fields:
            missing.append(f"participant_metadata.{field} (metadata missing)")

    # recording_metadata fields
    recording_fields = [
        "start_date_time_iso",
        "sampling_rate_hz",
        "recording_identifier",
    ]
    if (
        hasattr(datapoint, "recording_metadata")
        and getattr(datapoint, "recording_metadata") is not None
    ):
        meta = getattr(datapoint, "recording_metadata")
        for field in recording_fields:
            if field not in meta:
                missing.append(f"recording_metadata.{field} (missing)")
            elif meta[field] is None:
                missing.append(f"recording_metadata.{field} (is None)")
    else:
        for field in recording_fields:
            missing.append(f"recording_metadata.{field} (metadata missing)")

    # group_label check for expected structure
    if (
        hasattr(datapoint, "group_label")
        and getattr(datapoint, "group_label") is not None
    ):
        group_label = getattr(datapoint, "group_label")
        if not isinstance(group_label, dict):
            missing.append("group_label (not a dict)")
        else:
            for group_field in ["participant_id", "test", "trial"]:
                if group_field not in group_label:
                    missing.append(f"group_label.{group_field} (missing)")
                elif group_label[group_field] is None:
                    missing.append(f"group_label.{group_field} (is None)")
    else:
        for group_field in ["participant_id", "test", "trial"]:
            missing.append(f"group_label.{group_field} (missing group_label)")

    # sampling_rate_hz attribute check (already included above, but check numeric type)
    if hasattr(datapoint, "sampling_rate_hz"):
        sr = getattr(datapoint, "sampling_rate_hz")
        if sr is None:
            missing.append("sampling_rate_hz (is None)")
        elif not isinstance(sr, (int, float)):
            missing.append("sampling_rate_hz (not a number)")
    else:
        missing.append("sampling_rate_hz (attribute missing)")

    if verbose:
        if missing:
            print("Missing or incomplete datapoint fields:")
            for m in missing:
                print("  -", m)
        else:
            print("All required datapoint fields are present and complete.")

    return not bool(missing)
