from typing import Optional
import warnings
import pandas as pd
from typing_extensions import Self
from src.multigait.WS.base_ws import BaseWsDetector


class Ws(BaseWsDetector):
    """
    Walking speed calculator.

    This class computes walking speed (meters per second) from cadence
    (steps per minute) and stride length (meters) sampled per second.

    Walking speed per second is derived using the relationship:

        walking_speed_mps = (stride_length_m * cadence_spm) / (60 * 2)

    Attributes
    ----------
    walking_speed_per_sec_ : pd.DataFrame
        DataFrame containing walking speed in meters per second at one-second resolution.

    Other Parameters
    ----------------
    data : pd.DataFrame, optional
        Original dataframe containing sensor or gait event data. Stored for reference.
    initial_contacts : pd.DataFrame, optional
        Optional dataframe of detected initial foot contacts. Stored but not used in calculation.
    cadence_per_sec : pd.DataFrame
        DataFrame containing cadence at one-second resolution, with column "cadence_spm".
    stride_length_per_sec : pd.DataFrame
        DataFrame containing stride length at one-second resolution, with column "stride_length_m".
    sampling_rate_hz : float
        Sampling rate of the original signal in Hz. Stored but not used in computation.

    Notes
    -----
    Walking speed is computed as:

        walking_speed_mps = (stride_length_m * cadence_spm) / (60 * 2)

    - `cadence_spm / 60` converts steps/minute → steps/second
    - Division by 2 converts steps → strides (1 stride = 2 steps)
    - Therefore, `(stride_length_m * steps_per_second)` yields meters/second.
    """

    def __init__(self):
        pass

    def calculate(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        initial_contacts: Optional[pd.DataFrame] = None,
        cadence_per_sec: Optional[pd.DataFrame] = None,
        stride_length_per_sec: Optional[pd.DataFrame] = None,
        sampling_rate_hz: float = 100,
    ) -> Self:
        """
        Compute walking speed (m/s) at one-second resolution.

        Parameters
        ----------
        data : pd.DataFrame
           The original dataframe containing sensor or gait event data
           (not modified directly, but stored for reference).
        initial_contacts : pd.DataFrame, optional
           Optional dataframe of detected initial foot contacts. Stored
           but not used in the naive speed calculation.
        cadence_per_sec : pd.DataFrame, optional
           Dataframe containing cadence at one-second resolution, with
           column ``"cadence_spm"`` representing steps‐per‐minute.
           Must be provided.
        stride_length_per_sec : pd.DataFrame, optional
           Dataframe containing stride length at one-second resolution,
           with column ``"stride_length_m"`` representing meters per stride.
           Must be provided.
        sampling_rate_hz : float
           Sampling rate of the original signal (in Hz). Stored but not directly
           used in the naive computation.

        Returns
        -------
        Self
           The instance with the following attributes added:

           - ``walking_speed_per_sec_`` : pd.DataFrame
             A dataframe with a single column ``"walking_speed_mps"`` containing
             walking speed (meters/second) for each second.

        Raises
        ------
        ValueError
           If `cadence_per_sec` or `stride_length_per_sec` are not provided.

        Notes
        -----
        Walking speed is computed as:

           (stride_length_m * cadence_spm) / (60 * 2)

        - ``cadence_spm / 60`` converts steps/minute → steps/second
        - Division by 2 converts steps → strides (1 stride = 2 steps)
        - Therefore, `(stride_length_m * steps_per_second)` yields meters/second.
        """

        if cadence_per_sec is None:
            raise ValueError("cadence_per_sec must be provided for ws calculation")
        if stride_length_per_sec is None:
            raise ValueError(
                "stride_length_per_sec must be provided for ws calculation"
            )

        self.data = data
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        self.cadence_per_sec = cadence_per_sec
        self.stride_length_per_sec = stride_length_per_sec

        # Cadence and stride length must be the same length
        if len(cadence_per_sec) != len(stride_length_per_sec):
            warnings.warn(
                f"cadence_per_sec length ({len(cadence_per_sec)}) does not match "
                f"stride_length_per_sec length ({len(stride_length_per_sec)}). "
                "Walking speed will be aligned by index, but potential mismatch may affect accuracy.",
                UserWarning,
            )

        self.walking_speed_per_sec_ = (
            self.stride_length_per_sec["stride_length_m"]
            * self.cadence_per_sec["cadence_spm"]
            / (60 * 2)
        ).to_frame("walking_speed_mps")

        return self
