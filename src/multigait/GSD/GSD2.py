from typing_extensions import Self
import pandas as pd
import numpy as np
from typing import Literal
from src.multigait.GSD.utils.gravity_remove_butter import gravity_motion_butterworth
from src.multigait.GSD.utils.cwb import cwb
from src.multigait.GSD.base_gsd import BaseGsdDetector
from mobgap.data_transform import Resample, chain_transformers, ButterworthFilter


class HickeyGSD(BaseGsdDetector):
    """
    Implementation of the Gait Sequence Detection (GSD) algorithm by Hickey et al. (2017), adapted for wrist- and lowback-worn devices.

    The algorithm detects periods of walking by analyzing the acceleration signals.
    It assumes higher variability (SD) in the acceleration during movement compared to still periods.
    In the lowback versions the summary of the standard deviations of all axes are used for the variability detection and the vertical axis is used for the upright detection.
    The wrist versions use the norm of the acceleration with a similar rationale than the lower back.
    However, the upright threshold is used as maximum activity threshold for a wrist device rather than an upright threshold.
    This threshold is derived from identifying the 95th percentile of the signal from walking bouts in a sample of 108 participants with diverse conditions.

    This version contains a deviation from the original algorithm:
    - Walking bouts (WB) that are closer than 3 seconds can be merged according to the consensus for continuous walking bouts (CWB).
    - Users can optionally disable CWB (cwb=False) to retain micro-walking bouts (MWB), which represent very short, discrete walking events.
    - The pipeline is intended to be used with `cwb=True` to align with the consensus-based continuous walking detection, but micro-walking detection is available for research purposes.

    Other features:
    - Wrist and lowback versions differ in preprocessing and thresholds.
    - Acceleration signals are filtered, windowed (0.1 s), and analyzed for standard deviation and mean.
    - Detected bouts shorter than 0.5 s (minimum 2 strides) are discarded.

    Other features:
    - Wrist and lowback versions differ in preprocessing and thresholds.
    - Acceleration signals are filtered, windowed (0.1 s), and analyzed for standard deviation and mean.
    - Detected bouts shorter than 0.5 s (minimum 2 strides) are discarded when cwb is set to True.

    Parameters
    ----------
    version : Literal["original_lowback", "improved_lowback", "wrist"], default="wrist"
        Device placement and algorithm version. Only 'wrist' is fully supported in this release.
    cwb : bool, default=True
        If True, merges micro-walking bouts into continuous walking bouts according to the 3 s rule.
        If False, micro-walking bouts are retained without merging.

    Notes:
    Wrist versions:
    - Acceleration is preprocessed by removing gravity and calculating the norm of all three axes.
    - Thresholds for stillness and activity are based on empirical data from a diverse sample.
    - The algorithm operates on resampled data (default 100 Hz), hence it can be considered sample rate agnostic.
    - Continuous walking bouts can optionally be merged if separated by short breaks (≤3s).
    Lowback version:
    - Since we process the vertical axis to point downwards the, he threshold for identifying an upright position is sett to -0.77 instead of 0.77, as stated in the original publication
    - The standing threshold uses the mean of the original vertical axis without centering it, using the centered vertical axis always leads to the standingconddition being met.
    - The algorithm operates on resampled data (default 100 Hz), hence it can be considered sample rate agnostic.
    """

    def __init__(
        self,
        *,
        version: Literal["original_lowback", "improved_lowback", "wrist"] = "wrist",
        cwb: bool = True,
    ):
        """
        Initialise the HickeyGSD class.

        Parameters
        ----------
        version : str
            Only 'wrist' is supported in this release.
        cwb : bool
            If True, merges micro-walking bouts into continuous walking bouts.
        """

        if version not in ("original_lowback", "improved_lowback", "wrist"):
            raise ValueError(
                f"Unsupported version: {version}. Must be 'original_lowback', 'improved_lowback', or 'wrist'."
            )

        self.version = version
        self.cwb = cwb

        if self.version == "wrist":
            thresholdupright = 9.5  # in m/s^2 already
            thresholdstill = 0.1 * 9.81
        elif self.version == "improved_lowback":
            thresholdupright = -0.8 * 9.81
            thresholdstill = 0.17 * 9.81
        elif self.version == "original_lowback":
            thresholdupright = -0.77 * 9.81
            thresholdstill = 0.05 * 9.81

        self.data = None
        self.gs_list_ = None
        self.ThresholdStill = thresholdstill
        self.ThresholdUpright = thresholdupright

    def detect(
        self,
        data,
        *,
        sampling_rate_hz: float = 100,
        target_sampling_rate_hz: float = 100,
    ) -> Self:
        """
        Detect walking bouts from accelerometer data (wrist or lower-back).

        This method performs both preprocessing and detection of gait sequences in a single workflow.
        The procedure differs slightly depending on the device placement (`wrist` vs `lowback`).

        Preprocessing:

        - Wrist:
            1. Remove gravity from all three axes.
            2. Compute the norm of the acceleration vector.
            3. Optionally resample to the target sampling rate (default 100 Hz).

        - Lowback (original or improved):
            1. Invert vertical (IS) and mediolateral (ML) axes to standard orientation.
            2. Optionally resample to the target sampling rate (default 100 Hz).

        Detection:

        1. Center the acceleration signal:
            - Wrist: centered norm of acceleration.
            - Lowback: each axis centered individually.
        2. Apply a low-pass Butterworth filter (17 Hz cutoff) to remove high-frequency noise.
        3. Divide the signal into 0.1s windows and compute:
            - Standard deviation (SD) of each window.
                - Wrist: SD of norm.
                - Lowback: sum of SDs across axes.
            - Mean of vertical axis (lowback) or norm (wrist).
        4. Identify windows of movement based on threshold criteria:
            - Stillness threshold (`ThresholdStill`) applied to SD.
            - Upright/activity threshold (`ThresholdUpright`) applied to mean.
        5. Merge consecutive bouts separated by ≤2.25s (window units).
        6. Remove bouts shorter than 0.5s (require at least 2 strides).
        7. Optionally merge micro-walking bouts into continuous walking bouts (CWB) if `self.cwb=True`.

        Outputs:

        - `gs_list_` : pandas DataFrame
            Contains start and end indices of detected gait sequences.
            Index labeled as `gs_id`. Start and end are clipped to the length of the input signal.

        Notes:

        - Thresholds for wrist devices were derived empirically from a diverse sample of participants.
        - Lowback versions follow orientation conventions and specific thresholds per Hickey et al., 2017.
        - Sampling rate is assumed to be 100 Hz by default; resampling is optional and applied only if the data rate differs.
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.target_sampling_rate_hz = target_sampling_rate_hz
        cols = ["acc_is", "acc_ml", "acc_pa"]
        acc = self.data[cols]

        # We start with the preprocessing steps depending on the version
        if self.version in ["improved_lowback", "original_lowback"]:
            # vertical axis should point downwards
            # mediolateral axis should point to the right when looking at the subject from the front
            # anteroposterior axis should point forward (remain similar)
            acc_turned = pd.DataFrame()

            acc_turned["acc_is"] = -acc["acc_is"]
            acc_turned["acc_ml"] = -acc["acc_ml"]
            acc_turned["acc_pa"] = acc["acc_pa"]

            # Target sample rate is 100 which is similar to the sensor.
            # Added a check to see if the sensor sample rate is 100 and if it is then we don't resample
            if self.sampling_rate_hz != 100:
                filter_chain = [("resampling", Resample(self.target_sampling_rate_hz))]
                acc_turned = chain_transformers(
                    acc_turned, filter_chain, sampling_rate_hz=self.sampling_rate_hz
                )

            self.imu_preprocessed = acc_turned

        elif self.version in ["wrist"]:
            # removing gravity from the 3 axes using custom function
            acc_nograv = gravity_motion_butterworth(acc, sampling_rate_hz)

            # calculating norm of the acceleration
            acc_norm = np.linalg.norm(acc_nograv, axis=1)
            # converting to pandas DataFrame
            acc_norm = pd.DataFrame(acc_norm, columns=["acc_norm"])

            # Target sample rate is 100 which is similar to the sensor.
            # Added a check to see if the sensor sample rate is 100 and if it is then we don't resample
            if self.sampling_rate_hz != 100:
                filter_chain = [("resampling", Resample(self.target_sampling_rate_hz))]
                acc_norm = chain_transformers(
                    acc_norm, filter_chain, sampling_rate_hz=self.sampling_rate_hz
                )

            self.imu_preprocessed = acc_norm

        # Now we can proceed to the detection steps depending on the version
        if self.version == "wrist":
            data = self.imu_preprocessed

            # centering the norm acceleration
            acc_norm_mean = data.mean()
            acc_norm_centered = data - acc_norm_mean

            # Defining the window size which is 0.1s
            n = int(self.target_sampling_rate_hz / (self.target_sampling_rate_hz / 10))

            # Calculating the number of 0.1s windows present in the data
            win_num = int(len(acc_norm_centered) // n)

            # Performing a low pass butterworth filter on the data
            cutoff = 17
            # class instance
            filter_chain = [
                (
                    "butter",
                    ButterworthFilter(
                        order=2, cutoff_freq_hz=cutoff, filter_type="lowpass"
                    ),
                )
            ]

            # application to all corrected axes
            acc_filt = np.asarray(
                chain_transformers(
                    acc_norm_centered,
                    filter_chain,
                    sampling_rate_hz=self.sampling_rate_hz,
                )
            )

            # SD and mean calculation for all axes every 0.1s
            std_acc = np.zeros(win_num)
            mean_acc = np.zeros(win_num)

            for i in range(win_num):
                start_idx = int(i * n)
                end_idx = int((i + 1) * n)

                std_acc[i] = np.std(acc_filt[start_idx:end_idx])
                mean_acc[i] = np.mean(data[start_idx:end_idx])

            # Initialize the result array with zeros
            i_array_move_st_si = np.zeros(win_num)

            # Apply the conditions to each window
            for i in range(win_num):
                if (
                    std_acc[i] >= self.ThresholdStill
                    and mean_acc[i] <= self.ThresholdUpright
                ):
                    i_array_move_st_si[i] = 1

            # if i_array_move_st_si is all ones then the function should return a dataframe with the start and end of the signal!
            if i_array_move_st_si.sum() == win_num:
                self.gs_list_ = pd.DataFrame(
                    [[0, len(acc_norm_centered)]], columns=["start", "end"]
                )
                # Add an index "gs_id" that starts from 0
                self.gs_list_.index.name = "gs_id"
                return self

            # if i_array_move_st_si is all zeros then there is no walking and the function should return an empty dataframe
            if i_array_move_st_si.sum() == 0:
                self.gs_list_ = pd.DataFrame(columns=["start", "end"])
                # Add an index "gs_id" that starts from 0
                self.gs_list_.index.name = "gs_id"
                return self

            # Calculating starts and ends of walking
            # first and last elements should be 0 to identify transitions
            i_array_move_st_si[0] = 0
            i_array_move_st_si[-1] = 0

            # difference in array elements indicate start (1) and stop (-1)
            diffs = np.diff(i_array_move_st_si)
            bout_start_move_st_si = np.where(diffs == 1)[0] + 1
            bout_stop_move_st_si = np.where(diffs == -1)[0] + 1

            # Combine start and stop bouts into one array
            bout_array_move_st_si = np.column_stack(
                (bout_start_move_st_si, bout_stop_move_st_si)
            )

            # Initialize Difference Arrays
            betweenbbout_array_move_st_si = np.zeros(
                len(bout_array_move_st_si), dtype=int
            )

            # Set the first value of DifferenceArrayAMoveStSi to be similar to the first value of BoutArrayMoveStSi
            # the reason is that this represents the difference from the beginning of the signal to the first bout
            betweenbbout_array_move_st_si[0] = bout_array_move_st_si[0, 0]

            # Calculate the bout lengths
            boutlength_array_move_st_si = bout_stop_move_st_si - bout_start_move_st_si

            # Working out the differences between consecutive bouts
            for i in range(1, len(bout_array_move_st_si)):
                betweenbbout_array_move_st_si[i] = abs(
                    bout_stop_move_st_si[i - 1] - bout_start_move_st_si[i]
                )

            # Combine all the variables into one array, including start, stop, difference between bouts and bout length
            difference_array_move_st_si = np.column_stack(
                (
                    bout_start_move_st_si,
                    bout_stop_move_st_si,
                    betweenbbout_array_move_st_si,
                    boutlength_array_move_st_si,
                )
            )

            # Here we merge bouts which are 2.25s or less apart. Rationale is that two consequtive ICs
            # are expected to be from 0.25 to 2.25s appart so if two bouts have a smaller break than 2.25s then the break is walking
            # Using 22.5 due to scaling of windows to 0.1s so 2.25 seconds is 22.5 values
            i = 1  # Start from the second bout since we are merging with the previous one
            while i < len(difference_array_move_st_si):
                if difference_array_move_st_si[i, 2] <= 22.5:
                    # Merge current bout with the last one (index i-1)
                    difference_array_move_st_si[i - 1, 1] = difference_array_move_st_si[
                        i, 1
                    ]  # Update the stop time of the last bout
                    difference_array_move_st_si[i - 1, 3] += (
                        difference_array_move_st_si[i, 3]
                    )  # Combine bout lengths

                    # Remove the current row after merging
                    difference_array_move_st_si = np.delete(
                        difference_array_move_st_si, i, axis=0
                    )
                else:
                    i += 1  # Move to the next row if no merge is needed

            # According to consensus (Mob-D) a stride cannot be lower than 0.2s and if we need at least 2 strides to form a bout
            # we need to remove bouts that are shorter than 0.5s. This is in accordance with the original publication as well
            # Using 5 due to scaling of windows to 0.1s so half a second is 5 values
            difference_array_move_st_si = difference_array_move_st_si[
                difference_array_move_st_si[:, 3] > 5
            ]

            # Removing the 3rd column indicating the "pause" between bouts
            difference_array_move_st_si = np.delete(
                difference_array_move_st_si, 2, axis=1
            )

            # Converting back to samples
            difference_array_move_st_si = (difference_array_move_st_si * n).astype(int)

            # Create a pandas dataframe with the start and end of the gait sequences
            gs_list_ = pd.DataFrame(
                difference_array_move_st_si[:, [0, 1]], columns=["start", "end"]
            )

            # Add an index "gs_id" that starts from 0
            gs_list_.index.name = "gs_id"
            # Clipping start and end to be within limits of file
            gs_list_[["start", "end"]] = np.clip(
                gs_list_[["start", "end"]], 0, len(self.data)
            )

            # Creating Continuous Walking Bouts from micro walking bouts
            if self.cwb:
                gs_list_ = cwb(
                    gs_list_,
                    max_break_seconds=3,
                    sampling_rate=self.target_sampling_rate_hz,
                )

            self.gs_list_ = gs_list_

            return self

        elif self.version in ["improved_lowback", "original_lowback"]:
            data = self.imu_preprocessed

            acc_is = data["acc_is"]
            acc_ml = data["acc_ml"]
            acc_pa = data["acc_pa"]

            # Center the vertical (is) axis by subtracting its mean
            acc_is_mean = acc_is.mean()
            acc_is_centered = acc_is - acc_is_mean

            # Center the mediolateral (ml) axis by subtracting its mean
            acc_ml_mean = acc_ml.mean()
            acc_ml_centered = acc_ml - acc_ml_mean

            # Center the anteroposterior (pa) axis by subtracting its mean
            acc_pa_mean = acc_pa.mean()
            acc_pa_centered = acc_pa - acc_pa_mean

            # Defining the window size which is 0.1s
            n = int(self.target_sampling_rate_hz / (self.target_sampling_rate_hz / 10))

            # Calculating the number of 0.1s windows present in the data
            win_num = int(len(acc_is_centered) // n)

            # Performing a low pass butterworth filter on the data
            # The wintfilt function does not need to be in a different script as I will use the mogbap butter.
            cutoff = 17
            # class instance
            filter_chain = [
                (
                    "butter",
                    ButterworthFilter(
                        order=2, cutoff_freq_hz=cutoff, filter_type="lowpass"
                    ),
                )
            ]

            # application to all corrected axes
            acc_is_filt = np.asarray(
                chain_transformers(
                    acc_is_centered,
                    filter_chain,
                    sampling_rate_hz=self.sampling_rate_hz,
                )
            )
            acc_ml_filt = np.asarray(
                chain_transformers(
                    acc_ml_centered,
                    filter_chain,
                    sampling_rate_hz=self.sampling_rate_hz,
                )
            )
            acc_pa_filt = np.asarray(
                chain_transformers(
                    acc_pa_centered,
                    filter_chain,
                    sampling_rate_hz=self.sampling_rate_hz,
                )
            )

            # SD and mean calculation for all axes every 0.1s (window)
            std_acc_is = np.zeros(win_num)
            std_acc_ml = np.zeros(win_num)
            std_acc_pa = np.zeros(win_num)

            mean_acc_is = np.zeros(win_num)

            for i in range(win_num):
                start_idx = int(i * n)
                end_idx = int((i + 1) * n)

                std_acc_is[i] = np.std(acc_is_filt[start_idx:end_idx])
                std_acc_ml[i] = np.std(acc_ml_filt[start_idx:end_idx])
                std_acc_pa[i] = np.std(acc_pa_filt[start_idx:end_idx])

                mean_acc_is[i] = np.mean(acc_is[start_idx:end_idx])

            # Create a combined array of standard deviations
            std_acc = std_acc_is + std_acc_ml + std_acc_pa

            # Initialize the result array with zeros
            i_array_move_st_si = np.zeros(win_num)

            # Apply the conditions to each window. For the SD calculation, the centered and filter signal is used
            # For the standing threshold I use the raw signal
            for i in range(win_num):
                if (
                    std_acc[i] >= self.ThresholdStill
                    and mean_acc_is[i] <= self.ThresholdUpright
                ):
                    i_array_move_st_si[i] = 1

            # if i_array_move_st_si is all ones then the function should return a dataframe with the start and end of the signal!
            if i_array_move_st_si.sum() == win_num:
                self.gs_list_ = pd.DataFrame(
                    [[0, len(acc_is_centered)]], columns=["start", "end"]
                )
                # Add an index "gs_id" that starts from 0
                self.gs_list_.index.name = "gs_id"
                return self

            # if i_array_move_st_si is all zeros then there is no walking and the function should return an empty dataframe
            if i_array_move_st_si.sum() == 0:
                self.gs_list_ = pd.DataFrame(columns=["start", "end"])
                # Add an index "gs_id" that starts from 0
                self.gs_list_.index.name = "gs_id"
                return self

            # Calculating starts and ends of walking
            # first and last elements should be 0 to identify transitions
            i_array_move_st_si[0] = 0
            i_array_move_st_si[-1] = 0

            # difference in array elements indicate start (1) and stop (-1)
            diffs = np.diff(i_array_move_st_si)
            bout_start_move_st_si = np.where(diffs == 1)[0] + 1
            bout_stop_move_st_si = np.where(diffs == -1)[0] + 1

            # Combine start and stop bouts into one array
            bout_array_move_st_si = np.column_stack(
                (bout_start_move_st_si, bout_stop_move_st_si)
            )

            # Initialize Difference Arrays
            betweenbbout_array_move_st_si = np.zeros(
                len(bout_array_move_st_si), dtype=int
            )

            # Set the first value of DifferenceArrayAMoveStSi to be similar to the first value of BoutArrayMoveStSi
            # the reason is that this represents the difference from the beginning of the signal to the first bout
            betweenbbout_array_move_st_si[0] = bout_array_move_st_si[0, 0]

            # Calculate the bout lengths
            boutlength_array_move_st_si = bout_stop_move_st_si - bout_start_move_st_si

            # Working out the differences between consecutive bouts
            for i in range(1, len(bout_array_move_st_si)):
                betweenbbout_array_move_st_si[i] = abs(
                    bout_stop_move_st_si[i - 1] - bout_start_move_st_si[i]
                )

            # Combine all the variables into one array, including start, stop, difference between bouts and bout length
            difference_array_move_st_si = np.column_stack(
                (
                    bout_start_move_st_si,
                    bout_stop_move_st_si,
                    betweenbbout_array_move_st_si,
                    boutlength_array_move_st_si,
                )
            )

            # Here we merge bouts which are 2.25s or less apart. Rationale is that two consequtive ICs
            # are expected to be from 0.25 to 2.25s appart so if two bouts have a smaller break than 2.25s then the break is walking
            # Using 22.5 due to scaling of windows to 0.1s so 2.25 seconds is 22.5 values
            i = 1  # Start from the second bout since we are merging with the previous one
            while i < len(difference_array_move_st_si):
                if difference_array_move_st_si[i, 2] <= 22.5:
                    # Merge current bout with the last one (index i-1)
                    difference_array_move_st_si[i - 1, 1] = difference_array_move_st_si[
                        i, 1
                    ]  # Update the stop time of the last bout
                    difference_array_move_st_si[i - 1, 3] += (
                        difference_array_move_st_si[i, 3]
                    )  # Combine bout lengths

                    # Remove the current row after merging
                    difference_array_move_st_si = np.delete(
                        difference_array_move_st_si, i, axis=0
                    )
                else:
                    i += 1  # Move to the next row if no merge is needed

            # According to consensus (Mob-D) a stride cannot be lower than 0.2s and if we need at least 2 strides to form a bout
            # we need to remove bouts that are shorter than 0.5s. This is in accordance with the original publication as well
            # Using 5 due to scaling of windows to 0.1s so half a second is 5 values
            difference_array_move_st_si = difference_array_move_st_si[
                difference_array_move_st_si[:, 3] > 5
            ]

            # Removing the 3rd column indicating the "pause" between bouts
            difference_array_move_st_si = np.delete(
                difference_array_move_st_si, 2, axis=1
            )

            # Converting back to samples
            difference_array_move_st_si = (difference_array_move_st_si * n).astype(int)

            # Create a pandas dataframe with the start and end of the gait sequences
            gs_list_ = pd.DataFrame(
                difference_array_move_st_si[:, [0, 1]], columns=["start", "end"]
            )

            # Add an index "gs_id" that starts from 0
            gs_list_.index.name = "gs_id"
            # Clipping start and end to be within limits of file
            gs_list_[["start", "end"]] = np.clip(
                gs_list_[["start", "end"]], 0, len(self.data)
            )

            # Creating Continuous Walking Bouts from micro walking bouts
            if self.cwb:
                gs_list_ = cwb(
                    gs_list_,
                    max_break_seconds=3,
                    sampling_rate=self.target_sampling_rate_hz,
                )

            self.gs_list_ = gs_list_

            return self
