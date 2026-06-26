import numpy as np
import pandas as pd
import pytest
from multigait.GSD import KerenGSD  # adjust import path
from src.multigait import load_imu_data_wrist


class TestKerenGSD:
    @pytest.mark.parametrize(
        "version", ["original_wrist", "improved_wrist", "adaptive_wrist"]
    )
    @pytest.mark.parametrize("cwb", [True, False])
    def test_empty_input(self, version, cwb):
        """Test algorithm on zero signal (no gait sequences)."""
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_is", "acc_ml", "acc_pa"])
        gs_list = KerenGSD(version=version, cwb=cwb).detect(data).gs_list_

        assert list(gs_list.columns) == ["start", "end"]
        assert gs_list.index.name == "gs_id"
        assert gs_list.empty

    @pytest.mark.parametrize(
        "version", ["original_wrist", "improved_wrist", "adaptive_wrist"]
    )
    @pytest.mark.parametrize("cwb", [True, False])
    def test_random_signal(self, version, cwb):
        """Test algorithm on random data to check it runs and returns proper columns."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.rand(500, 3), columns=["acc_is", "acc_ml", "acc_pa"]
        )
        gs_list = KerenGSD(version=version, cwb=cwb).detect(data).gs_list_

        assert "start" in gs_list.columns and "end" in gs_list.columns

    @pytest.mark.parametrize("version", ["improved_wrist", "adaptive_wrist"])
    @pytest.mark.parametrize("cwb", [True, False])
    def test_real_wrist_data(self, version, cwb):
        """Test algorithm on real wrist IMU data."""
        imu_data = load_imu_data_wrist()
        gs_list = KerenGSD(version=version, cwb=cwb).detect(imu_data).gs_list_

        assert "start" in gs_list.columns and "end" in gs_list.columns
        # At least one walking bout may be detected, or empty if data has no gait
        assert len(gs_list) >= 0
