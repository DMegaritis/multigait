import pandas as pd
import numpy as np
from typing import Literal
from typing_extensions import Self
from scipy.signal import find_peaks
from mobgap.data_transform import chain_transformers, ButterworthFilter, Resample
from multigait.ICD.base_ic import BaseIcDetector


class DucharmeIC(BaseIcDetector):
    """
    Implementation of the Ducharme initial contact (IC) detection algorithm [1],
    for lowback devices (original and fine-tuned versions) and fine-tuned for wrist-worn accelerometer data.
    This is a peak detection method applied to a bandpass-filtered Euclidean norm of the acceleration signal.

        Workflow:
    1. Compute the Euclidean norm of the tri-axial accelerometer data.
    2. Detrend the signal by subtracting its mean.
    3. Downsample to 80 Hz, then apply a bandpass Butterworth filter (0.25–2.5 Hz).
    4. Detect peaks in the filtered signal that exceed the threshold.
    5. Upsample detected peak indices back to the original sampling rate.

    References
    ----------
    [1] Ducharme SW, Lim J, Busa MA, Aguiar EJ, Moore CC, Schuna JM Jr, Barreira TV, Staudenmayer J, Chipkin SR,
        Tudor-Locke C. A Transparent Method for Step Detection using an Acceleration Threshold.
        J Meas Phys Behav. 2021 Dec;4(4):311-320. doi: 10.1123/jmpb.2021-0011.
        Epub 2021 Oct 25. PMID: 36274923; PMCID: PMC9586317.

    Attributes
    ----------
    ic_list_ : pd.DataFrame
        DataFrame containing the indices of the detected initial contact events,
        resampled back to the original sampling rate.
    threshold : float
        Threshold for peak detection, adjusted from g to m/s².

    Notes
    -----
    - Threshold adjusted from g to m/s².
    - Resampling is performed because the original implementation used lower sampling rates.
    - Algorithm is sample-rate agnostic since all processing is performed at 80 Hz, and
      results are mapped back to the original rate.
    - Bandpass filtering compared to the original implementation in R: padding may differ and he beginning/end of the signal are more affected by filtering in Python, which is smoother.
      This can lead to first peak being under the threshold.
    - We inherit from BaseIcDetector to guarantee a consistent detect() contract, gain a built-in clone() helper for safe reuse,
        and integrate with tpcp-based pipelines and utilities.
    """

    ic_list_: pd.DataFrame

    def __init__(
        self,
        *,
        version: Literal["original_lowback", "improved_lowback", "wrist"] = "wrist",
    ) -> None:
        """
        Initialise the DucharmeIC detector.

        Parameters
        ----------
        version : Literal["original_lowback", "improved_lowback", "wrist"], optional
            Algorithm version.
        """

        if version not in ("original_lowback", "improved_lowback", "wrist"):
            raise ValueError(
                f"Unsupported version: {version}. Must be 'original_lowback', 'improved_lowback', or 'wrist'."
            )

        self.version = version

        if version == "wrist":
            self.threshold = 0.01 * 9.81
        elif version == "original_lowback":
            self.threshold = 0.0267 * 9.81
        elif version == "improved_lowback":
            self.threshold = 0.02 * 9.81

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float = 100) -> Self:
        """
        Process accelerometer data and detect initial contact (IC) events using the Ducharme algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            Input accelerometer data. The first three columns should contain the x, y, z axes.
        sampling_rate_hz : float
            Original sampling rate of the input signal in Hz.

        Returns
        -------
        Self
            Returns the instance with detected initial contacts stored in `ic_list_`.
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # 1. Euclidean norm of the data
        cols = ["acc_is", "acc_ml", "acc_pa"]
        acc_norm = np.linalg.norm(self.data[cols].values, axis=1)

        # 2. Detrend the signal by subtracting the mean
        acc_detr = acc_norm - np.mean(acc_norm)

        # 3. Bandpass Butterworth filtering
        # Because the original sampling rate was 80Hz (and 60Hz from another sensor),
        # here we downsample
        downsample = 80
        cutoff = (0.25, 2.5)
        filter_chain = [  # Resample to 80Hz for filtering with similar cutoffs as the original algo
            ("downsampling", Resample(downsample)),
            (
                "butter",
                ButterworthFilter(
                    order=4, cutoff_freq_hz=cutoff, filter_type="bandpass"
                ),
            ),
        ]

        acc_filt = chain_transformers(
            acc_detr, filter_chain, sampling_rate_hz=self.sampling_rate_hz
        )

        # 4. Peak detection
        peaks, _ = find_peaks(acc_filt, height=self.threshold)

        # Return the indices of the peaks
        final_ics = pd.DataFrame(columns=["ic"]).rename_axis(index="step_id")
        final_ics["ic"] = peaks

        # Upsample indices of peaks to the original sampling rate
        detected_ics_upsampled = (
            (final_ics * self.sampling_rate_hz / downsample).round().astype("int64")
        )

        self.ic_list_ = detected_ics_upsampled

        return self
