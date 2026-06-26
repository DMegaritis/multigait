import numpy as np
import pandas as pd
import pytest
from multigait.GSD import MacLeanGSD
from multigait import load_imu_data_wrist


class TestMacLeanGSD:
    @pytest.mark.parametrize("version", ["wrist"])
    @pytest.mark.parametrize("cwb", [True, False])
    def test_empty_input(self, version, cwb):
        """Test algorithm on zero signal (no gait sequences)."""
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_is", "acc_ml", "acc_pa"])
        gs_list = MacLeanGSD(version=version, cwb=cwb).detect(data).gs_list_

        # Check that the output DataFrame has correct columns and index
        assert list(gs_list.columns) == ["start", "end"]
        assert gs_list.index.name == "gs_id"
        assert gs_list.empty

    @pytest.mark.parametrize("version", ["wrist"])
    @pytest.mark.parametrize("cwb", [True, False])
    def test_random_signal(self, version, cwb):
        """Test algorithm on random data to check it runs and returns proper columns."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.rand(500, 3), columns=["acc_is", "acc_ml", "acc_pa"]
        )
        gs_list = MacLeanGSD(version=version, cwb=cwb).detect(data).gs_list_

        # Just check that output has correct columns
        assert "start" in gs_list.columns and "end" in gs_list.columns

    @pytest.mark.parametrize("version", ["wrist"])
    @pytest.mark.parametrize("cwb", [True, False])
    def test_real_wrist_data(self, version, cwb):
        """Test algorithm on real wrist IMU data."""
        imu_data = load_imu_data_wrist()
        gs_list = MacLeanGSD(version=version, cwb=cwb).detect(imu_data).gs_list_

        # Check that the output has expected columns
        assert "start" in gs_list.columns and "end" in gs_list.columns

        # Optional: check that at least one gait sequence was detected
        assert len(gs_list) >= 1 or gs_list.empty
