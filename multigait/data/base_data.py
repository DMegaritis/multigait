from typing import TypedDict
import pandas as pd
from tpcp import Dataset


class ParticipantMetadata(TypedDict):
    """
    Minimal participant metadata required by gait analysis algorithms.

    Attributes
    ----------
    height_m : float
        Participant's height in meters.
    sensor_height_m : float
        Height of the lower-back-mounted sensor in meters.
    foot_length_cm : float
        Length of the participant's feet in centimeters.
    leg_length_cm : float
        Length of the participant's legs in centimeters.
    arm_length_cm : float
        Length of the participant's arms in centimeters.
    """

    height_m: float
    sensor_height_m: float
    foot_length_cm: float
    leg_length_cm: float
    arm_length_cm: float


class BaseGaitDataset(Dataset):
    """
    Minimal base class for gait datasets.

    Provides a standard interface for pipelines consuming gait data, including
    participant metadata, sampling rate, and sensor data. Designed for datasets
    representing normal gait recordings.

    Parameters
    ----------
    %(general_dataset_args)s

    Attributes
    ----------
    sampling_rate_hz : float
        Sampling frequency of the dataset in Hertz.
    data_ss : pd.DataFrame
        Sensor signals or other per-sample data.
    participant_metadata : ParticipantMetadata
        Required participant information.

    Class Attributes
    ----------------
    UNITS : class
        Standard units for IMU sensor data in gait datasets.

    See Also
    --------
    %(dataset_see_also)s
    """

    class UNITS:
        """
        Standard IMU units for gait datasets.

        Attributes
        ----------
        acc : str
            Acceleration unit (default "ms^-2").
        gyr : str
            Gyroscope unit (default "deg/s").
        temp: str
            Temperature unit (default "degC").
        """

        acc: str = "ms^-2"
        gyr: str = "deg/s"
        temp: str = "degC"

    sampling_rate_hz: float
    data_ss: pd.DataFrame
    participant_metadata: ParticipantMetadata


__all__ = [
    "BaseGaitDataset",
    "ParticipantMetadata",
]
