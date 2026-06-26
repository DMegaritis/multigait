import warnings
import numpy as np
import pandas as pd
import pytest
from multigait.SL import BylemansSL


class TestBylemansSL:
    @pytest.mark.parametrize("version", ["wrist"])
    def test_invalid_version_parameter(self, version):
        """Passing an invalid version should raise ValueError."""
        with pytest.raises(ValueError):
            BylemansSL(version="invalid_version").calculate(
                data=pd.DataFrame(
                    np.zeros((100, 3)), columns=["acc_is", "acc_ml", "acc_pa"]
                ),
                initial_contacts=pd.DataFrame({"ic": np.arange(0, 100, 5)}),
                sampling_rate_hz=100,
            )

    @pytest.mark.parametrize("version", ["wrist"])
    def test_not_enough_ics(self, version):
        """With 0 or 1 initial contact, outputs should be all NaN and warnings raised."""
        data = pd.DataFrame(np.zeros((100, 3)), columns=["acc_is", "acc_ml", "acc_pa"])
        initial_contacts = pd.DataFrame({"ic": [0]})  # Only one IC
        sl = BylemansSL(version=version)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.calculate(data, initial_contacts, sampling_rate_hz=100)

        assert any(
            "Can not calculate step length" in str(warning.message) for warning in w
        )
        assert sl.step_length_per_sec_["step_length_m"].isna().all()
        assert sl.stride_length_per_sec_["stride_length_m"].isna().all()
        assert len(sl.raw_step_length_per_step_) == 0

    @pytest.mark.parametrize("version", ["wrist"])
    def test_unsorted_ics_warning(self, version):
        """Algorithm should sort unsorted ICs and raise a warning."""
        data = pd.DataFrame(np.zeros((100, 3)), columns=["acc_is", "acc_ml", "acc_pa"])
        initial_contacts = pd.DataFrame({"ic": [20, 5, 50]})  # Unsorted
        sl = BylemansSL(version=version)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sl.calculate(data, initial_contacts, sampling_rate_hz=100)

        assert any(
            "Initial contacts were not in ascending order" in str(warning.message)
            for warning in w
        )
        assert np.all(np.diff(sl.ic_list) > 0)

    @pytest.mark.parametrize("version", ["wrist"])
    @pytest.mark.parametrize("n_ics", [2, 3, 4])
    def test_small_n_ics(self, version, n_ics):
        """Test the algorithm with small number of ICs (minimal valid sequence)."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.rand(100, 3), columns=["acc_is", "acc_ml", "acc_pa"]
        )
        initial_contacts = pd.DataFrame({"ic": np.linspace(0, 99, n_ics, dtype=int)})
        sl = BylemansSL(version=version)
        sl.calculate(data, initial_contacts, sampling_rate_hz=100)

        assert len(sl.raw_step_length_per_step_) == n_ics - 1
        assert not sl.step_length_per_sec_["step_length_m"].isna().all()
        assert not sl.stride_length_per_sec_["stride_length_m"].isna().all()
        np.testing.assert_allclose(
            sl.stride_length_per_sec_["stride_length_m"].values,
            sl.step_length_per_sec_["step_length_m"].values * 2,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.1])
    def test_random_noise_effect(self, noise_std):
        """Adding white noise should not cause errors and outputs should remain plausible."""
        np.random.seed(42)
        data_clean = pd.DataFrame(
            np.random.rand(200, 3), columns=["acc_is", "acc_ml", "acc_pa"]
        )
        initial_contacts = pd.DataFrame({"ic": np.linspace(0, 199, 10, dtype=int)})
        sl_clean = BylemansSL(version="wrist").calculate(
            data_clean, initial_contacts, sampling_rate_hz=100
        )

        rng = np.random.default_rng(0)
        data_noisy = data_clean + rng.normal(scale=noise_std, size=data_clean.shape)
        sl_noisy = BylemansSL(version="wrist").calculate(
            data_noisy, initial_contacts, sampling_rate_hz=100
        )

        # Step lengths should not differ drastically
        diff = np.abs(
            sl_clean.step_length_per_sec_["step_length_m"]
            - sl_noisy.step_length_per_sec_["step_length_m"]
        ).mean()
        assert diff < 0.1

    def test_all_nan_with_zero_ic(self):
        """Passing 0 IC should produce all NaNs."""
        data = pd.DataFrame(np.zeros((100, 3)), columns=["acc_is", "acc_ml", "acc_pa"])
        initial_contacts = pd.DataFrame({"ic": []})
        sl = BylemansSL(version="wrist").calculate(
            data, initial_contacts, sampling_rate_hz=100
        )

        assert sl.step_length_per_sec_["step_length_m"].isna().all()
        assert sl.stride_length_per_sec_["stride_length_m"].isna().all()
        assert len(sl.raw_step_length_per_step_) == 0
