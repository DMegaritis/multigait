import warnings
from typing import Any
from multigait.GSD.utils.GSD1_utils import (
    find_pulse_trains,
    find_intersections,
    find_active_period_peak_threshold,
    NoActivePeriodsDetectedError,
    format_gait_sequences,
    combine_intervals,
)
from multigait.utils.data_conversions import seconds_to_samples
from multigait.GSD.base_gsd import BaseGsdDetector
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing_extensions import Self, Unpack

from mobgap.data_transform import (
    CwtFilter,
    EpflDedriftedGaitFilter,
    EpflGaitFilter,
    GaussianFilter,
    Pad,
    Resample,
    SavgolFilter,
    chain_transformers,
)


class _BaseIonescuGSD(BaseGsdDetector):
    """
    Base class for the Ionescu GSD algorithm.

    This class implements the general workflow for detecting gait sequences from accelerometer data,
    including preprocessing, candidate step detection, padding, filtering, and merging of sequences.
    Subclasses (e.g., IonescuGSD) implement device-specific logic such as thresholding and filter parameters.

    This implementation has been fine-tuned for wrist-worn devices, adapting filters, padding, and step detection
    thresholds for wrist accelerometer signals. In addition, the logic for merging walking bouts has been customised.

    Attributes
    ----------
    min_n_steps : int
        Minimum number of steps required for a gait sequence to be considered valid.
    max_gap_s : float
        Maximum allowed gap in seconds between consecutive gait sequences to merge them.
        Based on the original method it is 3.5s (initial THd = 3.5 s).
        Here we merge by < 3 s according to WB rules.
    min_step_margin_s : float
        Margin in seconds used when combining peaks into pulse trains.
    padding : float
        Fraction of mean step duration used to extend each gait sequence before and after.
    cwb : bool
        Whether to merge micro walking bouts into continuous walking bouts. Note: in order for the GSD algorithms
        to be used with the pipeline the cwb should be True (default).
    """

    min_n_steps: int
    active_signal_threshold: float
    max_gap_s: float
    min_step_margin_s: float
    padding: float

    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(
        self,
        *,
        version: str = "wrist",
        active_signal_threshold: float = 0.31,
        active_signal_fallback_threshold: float = 0.4,
        percentile: int = 31,
        min_n_steps: int = 4,
        max_gap_s: float = 3,
        min_step_margin_s: float = 1.5,
        padding: float = 0.75,
        cwb: bool = True,
    ) -> None:

        if version not in ("wrist", "wrist_adaptive"):
            raise ValueError(
                f"Unsupported version: {version}. Must be 'wrist' or 'wrist_adaptive'."
            )

        self.version = version
        self.active_signal_threshold = active_signal_threshold
        self.active_signal_fallback_threshold = active_signal_fallback_threshold
        self.percentile = percentile
        self.min_n_steps = min_n_steps
        self.max_gap_s = max_gap_s
        self.min_step_margin_s = min_step_margin_s
        self.padding = padding
        self.cwb = cwb

        self.active_signal_threshold_m_s2 = active_signal_threshold * 9.81
        self.active_signal_fallback_threshold_m_s2 = (
            active_signal_fallback_threshold * 9.81
        )

        super().__init__()

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float = 100,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        self.data_len = len(data)
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        cols = ["acc_is", "acc_ml", "acc_pa"]
        acc = self.data[cols]

        if not self.min_n_steps >= 1:
            raise ValueError("min_n_steps must be at least 1")

        # Signal vector magnitude
        acc_norm = np.linalg.norm(acc, axis=1)

        # Peaks are in samples based on internal sampling rate
        min_peaks, max_peaks = self._detect_candidate_ics(acc_norm, sampling_rate_hz)

        # Combine steps detected by the maxima and minima
        allowed_distance_between_peaks = seconds_to_samples(
            self.min_step_margin_s, self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        )
        step_margin = seconds_to_samples(
            self.min_step_margin_s, self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        )

        gs_from_max = find_pulse_trains(
            max_peaks, allowed_distance_between_peaks, step_margin
        )
        gs_from_min = find_pulse_trains(
            min_peaks, allowed_distance_between_peaks, step_margin
        )

        # Combine the gs from the maxima and minima
        combined_final = find_intersections(gs_from_max, gs_from_min)

        # Check if all gs removed and return empty df
        if combined_final.size == 0:
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            # Add an index "gs_id" that starts from 0
            self.gs_list_.index.name = "gs_id"
            return self

        # Find all max_peaks within each final gs
        steps_per_gs = [
            [x for x in max_peaks if gs[0] <= x <= gs[1]] for gs in combined_final
        ]
        n_steps_per_gs = np.array([len(steps) for steps in steps_per_gs])
        # It can happen that we only have one step in a gs, in this case we can not calculate the mean step time
        # Numpy will output NaN and throw a warning.
        # We don't want ot see the warning, so we suppress it.
        # GSs that don't have enough steps will be removed later anyway.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_step_times = np.array(
                [np.mean(np.diff(steps)) for steps in steps_per_gs]
            )
            # Pad each gs by padding*mean_step_times before and after
            combined_final[:, 0] = np.fix(
                combined_final[:, 0] - self.padding * mean_step_times
            )
            combined_final[:, 1] = np.fix(
                combined_final[:, 1] + self.padding * mean_step_times
            )

        # Filter again by number of steps, remove any gs with too few steps
        combined_final = combined_final[n_steps_per_gs >= self.min_n_steps]

        # Check if all gs removed and return empty df
        if combined_final.size == 0:
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            # Add an index "gs_id" that starts from 0
            self.gs_list_.index.name = "gs_id"
            return self

        # Here we call the merging function if the cwb is True by merging bouts which are closer than 3 seconds
        # If cwb is False then we call this step but with the max_gap_s as 0.
        # Merge gs if time (in seconds) between consecutive gs is smaller than max_gap_s
        if self.cwb:
            combined_final = combine_intervals(
                combined_final,
                seconds_to_samples(
                    self.max_gap_s, self._INTERNAL_FILTER_SAMPLING_RATE_HZ
                ),
            )
        else:
            combined_final = combine_intervals(
                combined_final,
                seconds_to_samples(0, self._INTERNAL_FILTER_SAMPLING_RATE_HZ),
            )

        # Convert back to original sampling rate
        combined_final = (
            combined_final * sampling_rate_hz / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        )

        # Cap the start and the end of the signal using clip, in case padding extended any gs past the signal length
        combined_final = np.clip(combined_final, 0, len(acc))

        # Compile the df
        self.gs_list_ = format_gait_sequences(
            pd.DataFrame(combined_final, columns=["start", "end"])
        )

        return self


class IonescuGSD(_BaseIonescuGSD):
    """
     Implementation of the Gait Sequence Detection (GSD) algorithm by Paraschiv-Ionescu et al. [1], fine-tuned for wrist-worn accelerometers.

     The algorithm detects individual steps and combines them into gait sequences.
     Two modes are supported:
     - "wrist": fixed threshold detection based on acceleration magnitude.
     - "wrist_adaptive": adaptive threshold detection based on periods of activity in the signal.

     Preprocessing:
     - Signal is resampled to an internal 40 Hz rate.
     - Wrist data is filtered using Savitzky-Golay filters, EPFL dedrifted gait filter, continuous wavelet transform (CWT), and optional Gaussian smoothing.
     - The filtered signal is used to detect peaks corresponding to initial contacts (ICs).

     Detection:
     - Candidate ICs are detected as minima and maxima of the filtered signal above the threshold.
     - Peaks are grouped into pulse trains, representing candidate gait sequences (GS).
     - GS are padded according to mean step times and filtered by minimum number of steps.
     - GS that are close in time (below `max_gap_s`) are merged.
     - Final start/end indices are converted back to original sampling rate and clipped to signal length.

     Continuous Walking Bouts (CWB):
     - Optional merging of micro walking bouts into continuous walking bouts using `cwb()` function.
     - Default maximum break between bouts is 3 seconds.

     Notes:
     - The wrist mode uses a fixed threshold for step detection (converted to m/s² internally).
     - The wrist_adaptive mode calculates thresholds dynamically from detected active periods.
     - Resampling ensures that the algorithm is sample-rate agnostic.
     - The provided thresholds for both versions have been tuned in multimorbid populations.

     .. [1] Paraschiv-Ionescu, A, et al. "Locomotion and cadence detection using a single trunk-fixed accelerometer:
    validity for children with cerebral palsy in daily life-like conditions." Journal of neuroengineering and
    rehabilitation 16.1 (2019): 1-11.
    """

    filtered_signal_: np.ndarray
    threshold_: float
    adaptive_threshold_success_: bool

    def __init__(
        self,
        *,
        version: str = "wrist",
        active_signal_threshold: float = 0.31,
        active_signal_fallback_threshold: float = 0.4,
        percentile: int = 31,
        min_n_steps: int = 4,
        max_gap_s: float = 3,
        min_step_margin_s: float = 1.5,
        padding: float = 0.75,
        cwb: bool = True,
    ) -> None:
        super().__init__(
            version=version,
            active_signal_threshold=active_signal_threshold,
            active_signal_fallback_threshold=active_signal_fallback_threshold,
            percentile=percentile,
            min_n_steps=min_n_steps,
            max_gap_s=max_gap_s,
            min_step_margin_s=min_step_margin_s,
            padding=padding,
            cwb=cwb,
        )

        # Keep convenience references that do not shadow parameter names:
        self.version = version
        self.cwb = cwb
        self.percentile = percentile

    def _detect_candidate_ics(
        self, acc_norm: np.ndarray, sampling_rate_hz: float
    ) -> tuple:
        """Find candidate ICs based on version (wrist or wrist_adaptive)."""
        # Common filters
        cwt = CwtFilter(wavelet="gaus2", center_frequency_hz=1.2)

        # Savgol filters
        if self.version == "wrist":
            savgol_win_samples = 3
            savgol_1 = SavgolFilter(
                window_length_s=savgol_win_samples
                / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
                polyorder_rel=1 / savgol_win_samples,
            )
            savgol_2 = savgol_1.clone()
        else:
            # wrist_adaptive uses original two-stage savgol filters
            savgol_1 = SavgolFilter(
                window_length_s=21 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
                polyorder_rel=7 / 21,
            )
            savgol_2 = SavgolFilter(
                window_length_s=11 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ,
                polyorder_rel=5 / 11,
            )

        # Padding
        n_coefficients = len(EpflGaitFilter().coefficients[0])
        len_pad_s = 4 * n_coefficients / self._INTERNAL_FILTER_SAMPLING_RATE_HZ
        padding = Pad(pad_len_s=len_pad_s, mode="wrap")

        if self.version == "wrist":
            filter_chain = [
                ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
                ("padding", padding),
                ("savgol_1", savgol_1),
                ("epfl_gait_filter", EpflDedriftedGaitFilter()),
                ("cwt", cwt),
                ("savol_2", savgol_2),
                ("padding_remove", padding.get_inverse_transformer()),
            ]
            signal = chain_transformers(
                acc_norm, filter_chain, sampling_rate_hz=sampling_rate_hz
            )
            self.filtered_signal_ = signal
            active_peak_threshold = self.active_signal_threshold_m_s2

        elif self.version == "wrist_adaptive":
            filter_chain = [
                ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
                ("padding", padding),
                ("savgol_1", savgol_1),
                ("epfl_gait_filter", EpflDedriftedGaitFilter()),
                ("cwt_1", cwt),
                ("savol_2", savgol_2),
                ("cwt_2", cwt),
                (
                    "gaussian_1",
                    GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ),
                ),
                (
                    "gaussian_2",
                    GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ),
                ),
                (
                    "gaussian_3",
                    GaussianFilter(sigma_s=3 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ),
                ),
                (
                    "gaussian_4",
                    GaussianFilter(sigma_s=2 / self._INTERNAL_FILTER_SAMPLING_RATE_HZ),
                ),
                ("padding_remove", padding.get_inverse_transformer()),
            ]
            acc_filtered = chain_transformers(
                acc_norm, filter_chain, sampling_rate_hz=sampling_rate_hz
            )

            try:
                active_peak_threshold = find_active_period_peak_threshold(
                    self,
                    signal=acc_filtered,
                    hilbert_window_size=seconds_to_samples(
                        1, self._INTERNAL_FILTER_SAMPLING_RATE_HZ
                    ),
                    min_active_period_duration=seconds_to_samples(
                        3, self._INTERNAL_FILTER_SAMPLING_RATE_HZ
                    ),
                )
                signal = acc_filtered
                self.adaptive_threshold_success_ = True
            except NoActivePeriodsDetectedError:
                self.adaptive_threshold_success_ = False
                warnings.warn(
                    "No active periods detected, using fallback threshold", stacklevel=1
                )
                active_peak_threshold = self.active_signal_fallback_threshold_m_s2

                fallback_filter_chain = [
                    ("resampling", Resample(self._INTERNAL_FILTER_SAMPLING_RATE_HZ)),
                    ("padding", padding),
                    ("savgol_1", savgol_1),
                    ("epfl_gait_filter", EpflDedriftedGaitFilter()),
                    ("cwt_1", cwt),
                    ("savol_2", savgol_2),
                    ("padding_remove", padding.get_inverse_transformer()),
                ]
                signal = chain_transformers(
                    acc_norm, fallback_filter_chain, sampling_rate_hz=sampling_rate_hz
                )

            self.filtered_signal_ = signal

        # Find extrema
        return find_peaks(-signal, height=active_peak_threshold)[0], find_peaks(
            signal, height=active_peak_threshold
        )[0]
