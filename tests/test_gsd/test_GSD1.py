import numpy as np
import pandas as pd
import pytest
from multigait.GSD.GSD1 import IonescuGSD
from multigait.utils.data_loader import load_imu_data_wrist


class TestIonescuGSD:
    @pytest.mark.parametrize("version", ["wrist", "wrist_adaptive"])
    def test_empty_input(self, version):
        data = pd.DataFrame(np.zeros((1000, 3)), columns=["acc_is", "acc_ml", "acc_pa"])
        gs_list = (
            IonescuGSD(version=version).detect(data, sampling_rate_hz=100).gs_list_
        )

        # Only check columns and index name
        assert list(gs_list.columns) == ["start", "end"]
        assert gs_list.index.name == "gs_id"
        assert gs_list.empty

    @pytest.mark.parametrize("version", ["wrist", "wrist_adaptive"])
    def test_random_signal(self, version):
        """Test algorithm on random data to check it runs and returns proper columns."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.rand(500, 3), columns=["acc_is", "acc_ml", "acc_pa"]
        )
        gs_list = (
            IonescuGSD(version=version).detect(data, sampling_rate_hz=100).gs_list_
        )

        # Just check output has correct columns
        assert "start" in gs_list.columns and "end" in gs_list.columns

    @pytest.mark.parametrize("version", ["wrist"])
    def test_real_wrist_data(self, version):
        """Test algorithm on real wrist IMU data."""
        imu_data = load_imu_data_wrist()
        gs_list = (
            IonescuGSD(version=version).detect(imu_data, sampling_rate_hz=100).gs_list_
        )

        # Check that the output has the expected columns
        assert "start" in gs_list.columns and "end" in gs_list.columns

        # Optional: check that at least one gait sequence was detected
        assert len(gs_list) >= 1
