import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from multigait.CAD.cad import Cadence


class TestCadence:
    @pytest.mark.parametrize("sampling_rate_hz", [10.0, 20.0, 40.0])
    @pytest.mark.parametrize("fixed_step_size", [5, 10, 20])
    def test_naive(self, sampling_rate_hz, fixed_step_size):
        n_samples = 100
        data = pd.DataFrame(
            np.zeros((n_samples, 3)), columns=["acc_x", "acc_y", "acc_z"]
        )
        initial_contacts = pd.DataFrame(
            {"ic": np.arange(0, n_samples + 1, fixed_step_size)}
        )
        data = data.iloc[: initial_contacts["ic"].iloc[-1]]

        cad = Cadence().calculate(
            data, initial_contacts=initial_contacts, sampling_rate_hz=sampling_rate_hz
        )
        cadence = cad.cadence_per_sec_

        expected_index = pd.Index(
            np.arange(
                0.5 * sampling_rate_hz,
                n_samples + 0.5 * sampling_rate_hz,
                sampling_rate_hz,
            ),
            name="sec_center_samples",
        ).astype("int64")

        expected_output = (
            np.ones(len(expected_index)) * 1 / (fixed_step_size / sampling_rate_hz) * 60
        )

        assert len(cadence) == np.ceil(len(data) / sampling_rate_hz)
        assert_frame_equal(
            cadence,
            pd.DataFrame({"cadence_spm": expected_output}, index=expected_index),
        )

    def test_empty_ics_warns_and_returns_nan(self):
        n_samples = 50
        data = pd.DataFrame(
            np.zeros((n_samples, 3)), columns=["acc_x", "acc_y", "acc_z"]
        )
        initial_contacts = pd.DataFrame({"ic": []})
        sampling_rate_hz = 40.0

        with pytest.warns(UserWarning):
            cad = Cadence().calculate(
                data,
                initial_contacts=initial_contacts,
                sampling_rate_hz=sampling_rate_hz,
            )

        assert cad.cadence_per_sec_["cadence_spm"].isna().all()
        assert len(cad.cadence_per_sec_) == np.ceil(len(data) / sampling_rate_hz)

    def test_single_ic_warns_and_returns_nan(self):
        data = pd.DataFrame(np.zeros((50, 3)), columns=["acc_x", "acc_y", "acc_z"])
        initial_contacts = pd.DataFrame({"ic": [10]})
        sampling_rate_hz = 40.0

        with pytest.warns(UserWarning):
            cad = Cadence().calculate(
                data,
                initial_contacts=initial_contacts,
                sampling_rate_hz=sampling_rate_hz,
            )

        assert cad.cadence_per_sec_["cadence_spm"].isna().all()

    def test_non_sorted_ics_are_warned_and_sorted(self):
        data = pd.DataFrame(np.zeros((50, 3)), columns=["acc_x", "acc_y", "acc_z"])
        initial_contacts = pd.DataFrame({"ic": [20, 5, 15]})
        sampling_rate_hz = 40.0

        with pytest.warns(UserWarning):
            cad = Cadence().calculate(
                data,
                initial_contacts=initial_contacts,
                sampling_rate_hz=sampling_rate_hz,
            )

        assert cad.cadence_per_sec_ is not None
