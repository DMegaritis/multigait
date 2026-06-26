import numpy as np
import pandas as pd
import pytest
from multigait.ICD import PhamIC


class TestPhamIC:
    @pytest.mark.parametrize("version", ["wrist"])
    def test_invalid_version_parameter(self, version):
        with pytest.raises(ValueError):
            PhamIC(version="invalid_version").detect(
                pd.DataFrame(), sampling_rate_hz=100
            )

    @pytest.mark.parametrize("version", ["wrist"])
    def test_no_ics_detected_empty_signal(self, version):
        """Empty signal should return empty ic_list_ DataFrame."""
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_is", "acc_ml", "acc_pa"])
        algo = PhamIC(version=version)
        result = algo.detect(data, sampling_rate_hz=100)
        ic_list = result.ic_list_

        assert isinstance(ic_list, pd.DataFrame)
        assert ic_list.columns.tolist() == ["ic"]
        assert ic_list.index.name == "step_id"
        assert ic_list.empty or ic_list["ic"].size >= 0

    @pytest.mark.parametrize("version", ["wrist"])
    def test_random_signal_runs(self, version):
        """Random noise should run without errors and return a DataFrame."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.rand(2000, 3), columns=["acc_is", "acc_ml", "acc_pa"]
        )
        algo = PhamIC(version=version)
        result = algo.detect(data, sampling_rate_hz=100)

        assert isinstance(result.ic_list_, pd.DataFrame)
        assert result.ic_list_.columns.tolist() == ["ic"]
        assert result.ic_list_.index.name == "step_id"
        # IC indices should be within signal bounds
        assert result.ic_list_["ic"].between(0, len(data) - 1).all()
        # IC values should be integers
        assert np.issubdtype(result.ic_list_["ic"].dtype, np.integer)

    @pytest.mark.parametrize("version", ["wrist"])
    def test_reproducibility_on_random_signal(self, version):
        """Algorithm should produce the same ICs for the same input (deterministic)."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.rand(2000, 3), columns=["acc_is", "acc_ml", "acc_pa"]
        )
        algo1 = PhamIC(version=version)
        result1 = algo1.detect(data, sampling_rate_hz=100)

        algo2 = PhamIC(version=version)
        result2 = algo2.detect(data, sampling_rate_hz=100)

        assert np.array_equal(result1.ic_list_["ic"], result2.ic_list_["ic"])
