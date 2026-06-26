import warnings
from typing import Any
import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from mobgap.data_transform import HampelFilter
from multigait.CAD.utils.cad_utils import smooth_and_bin_steps
from multigait.CAD.base_cad import BaseCadDetector


class Cadence(BaseCadDetector):
    """
    Compute per-second cadence from initial contact (IC) timestamps.

    This detector calculates cadence in steps per minute for each second of the
    measurement period. It assumes that initial contacts are sorted chronologically,
    with the first IC representing the start and the last IC the end of recording.

    Missing ICs are linearly interpolated for gaps smaller than `max_interpolation_gap_s`;
    larger gaps remain NaN. Per-second cadence values are smoothed using a Hampel filter
    to reduce outliers and noise.

    Inherits from `BaseCadDetector`, providing a `clone()` implementation for repeated
    use on fresh instances.

    Parameters
    ----------
    max_interpolation_gap_s : int
        Maximum gap (in seconds) for which linear interpolation is applied.
    hampel_window : int
        Half-window size of the Hampel filter for outlier smoothing.
    n_sigma : float
        Threshold in standard deviations for Hampel filter outlier detection.

    Attributes
    ----------
    cadence_per_sec_ : pd.DataFrame
        Per-second cadence (steps/min) with index corresponding to the center of each second
        in sample units (`sec_center_samples`).
    """

    max_interpolation_gap_s: int

    def __init__(
        self,
        *,
        max_interpolation_gap_s: int = 3,
        hampel_window: int = 2,
        n_sigma: float = 3.0,
    ) -> None:
        self.max_interpolation_gap_s = max_interpolation_gap_s
        self.hampel_window = hampel_window
        self.n_sigma = n_sigma

        # Initialising the HampelFilter (it's fine to create here, but tpcp needs the raw params stored as attributes)
        self.step_smoothing_filter = HampelFilter(
            half_window_size=hampel_window, n_sigmas=n_sigma
        )

    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: pd.DataFrame,
        sampling_rate_hz: float = 100,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Calculate cadence from initial contact timestamps.

        Parameters
        ----------
        data : pd.DataFrame
            The raw time series or measurement data.
        initial_contacts : pd.DataFrame
            DataFrame containing the initial contact timestamps under column 'ic'.
        sampling_rate_hz : float
            Sampling frequency of the data in Hertz.
        kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        Self
            The instance itself with `cadence_per_second` attribute populated.
        """
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        # Accessing ICs
        initial_contacts = initial_contacts["ic"]
        # Checking that ICs are not empty
        if len(initial_contacts) == 0:
            warnings.warn(
                "No initial contacts provided. Cadence will be NaN", stacklevel=2
            )

        # Checking if ICs are sorted and sorting if necessary
        if not initial_contacts.is_monotonic_increasing:
            warnings.warn(
                "ICs must be sorted in ascending order. Automatically resorting to ascending order.",
                stacklevel=2,
            )
            initial_contacts = initial_contacts.sort_values().reset_index(drop=True)

        initial_contacts_in_seconds = initial_contacts / sampling_rate_hz
        n_secs = len(data) / sampling_rate_hz
        sec_centers = 0.5 + np.arange(int(np.ceil(n_secs)))

        # Cadence calculation per second
        ic_times = (
            initial_contacts_in_seconds.to_numpy()
            if isinstance(initial_contacts_in_seconds, pd.Series)
            else initial_contacts_in_seconds
        )

        if len(ic_times) <= 1:
            warnings.warn(
                "Cannot calculate cadence with zero or one ICs.",
                stacklevel=3,
            )
            cadence_spm = np.full(len(sec_centers), np.nan)
        else:
            # Step times between consecutive initial contacts
            step_durations = np.diff(ic_times)
            # Repeat last step to match number of initial contacts
            step_durations = np.concatenate([step_durations, step_durations[-1:]])

            # Smooth & interpolate to per-second step times
            step_time_per_sec_smooth = smooth_and_bin_steps(
                step_times=ic_times,
                step_values=step_durations,
                second_midpoints=sec_centers,
                max_gap_s=self.max_interpolation_gap_s,
                filter_obj=self.step_smoothing_filter,
            )

            # Convert step time per second to cadence in steps per minute
            cadence_spm = 60.0 / step_time_per_sec_smooth

        self.cadence_per_sec_ = pd.DataFrame(
            {"cadence_spm": cadence_spm},
            index=np.round(sec_centers * sampling_rate_hz).astype("int64"),
        ).rename_axis(index="sec_center_samples")

        return self


class CadenceSimple:
    """
    Lightweight stride-based average cadence estimator.

    Computes the mean cadence in steps per minute (spm) from initial contacts (ICs)
    using a stride-based approach:

        cadence_spm = 2 * mean(60 / strideDurations)

    where `strideDurations` are the durations of consecutive strides (every second IC),
    i.e., IC1→IC3, IC2→IC4, etc.

    This method is slightly more robust than simple IC-count/duration approaches,
    while remaining lightweight and computationally inexpensive.

    Notes
    -----
    - This is not to be used with the pipeline.
    - Requires at least 3 ICs to compute one full stride; otherwise, returns NaN.

    Attributes
    ----------
    cadence_spm_ : float
        Computed average cadence in steps per minute.
    """

    def __init__(self) -> None:
        return

    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: pd.DataFrame,
        sampling_rate_hz: float,
    ) -> float:
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        ic_times = initial_contacts["ic"].to_numpy() / sampling_rate_hz
        n_ics = len(ic_times)

        # Need at least 3 ICs to form one full stride
        if n_ics < 3:
            warnings.warn(
                "Not enough ICs to calculate stride-based cadence. Returning NaN.",
                stacklevel=2,
            )
            self.cadence_spm_ = float("nan")
            return self.cadence_spm_

        # Stride durations: time between every 2nd IC (IC1->IC3, IC2->IC4, etc.)
        stride_durations = ic_times[2:] - ic_times[:-2]

        # Average stride-based cadence, converted to steps/min
        self.cadence_spm_ = float(2.0 * np.mean(60.0 / stride_durations))

        return self.cadence_spm_
