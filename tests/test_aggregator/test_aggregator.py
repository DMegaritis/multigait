import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from multigait.aggregation import GenericAggregator


@pytest.fixture
def example_dmo_data():
    # create minimal dummy data matching GenericAggregator.INPUT_COLUMNS
    n = 5
    data = pd.DataFrame(
        {
            "participant_id": ["p1"] * n,
            "measurement_date": ["2025-12-10"] * n,
            "wb_id": [f"wb{i}" for i in range(n)],
            "duration_s": np.arange(5, 5 + n),
            "n_raw_initial_contacts": np.arange(10, 10 + n),
            "walking_speed_mps": np.linspace(0.8, 1.2, n),
            "stride_length_m": np.linspace(0.5, 0.7, n),
            "cadence_spm": np.linspace(90, 110, n),
            "stride_duration_s": np.linspace(0.5, 0.6, n),
            "stride_duration_s_cv": np.random.rand(n),
            "stride_duration_s_rmssd": np.random.rand(n),
            "walking_speed_mps_cv": np.random.rand(n),
            "walking_speed_mps_rmssd": np.random.rand(n),
            "stride_length_m_cv": np.random.rand(n),
            "stride_length_m_rmssd": np.random.rand(n),
            "cadence_spm_cv": np.random.rand(n),
            "cadence_spm_rmssd": np.random.rand(n),
            "alpha": np.random.rand(n),
        }
    )
    data.set_index(["participant_id", "measurement_date", "wb_id"], inplace=True)
    return data


@pytest.fixture
def dummy_dmo_data_mask(example_dmo_data):
    return example_dmo_data.astype(bool)


class TestGenericAggregator:
    """Simplified tests for GenericAggregator."""

    def test_aggregate_runs(self, example_dmo_data):
        agg = GenericAggregator(
            **GenericAggregator.PredefinedParameters.single_day
        ).aggregate(example_dmo_data)
        assert hasattr(agg, "aggregated_data_")
        assert hasattr(agg, "filtered_wb_dmos_")
        # Ensure some output values exist
        assert not agg.aggregated_data_.empty

    def test_input_not_modified(self, example_dmo_data, dummy_dmo_data_mask):
        data = example_dmo_data.copy()
        mask = dummy_dmo_data_mask.copy()
        agg = GenericAggregator(
            **GenericAggregator.PredefinedParameters.single_day
        ).aggregate(data, wb_dmos_mask=mask)
        assert_frame_equal(data, agg.wb_dmos)
        assert_frame_equal(mask, agg.wb_dmos_mask)

    def test_raise_error_on_wrong_data(self):
        with pytest.raises(ValueError):
            GenericAggregator(
                **GenericAggregator.PredefinedParameters.single_day
            ).aggregate(pd.DataFrame(np.random.rand(3, 3)))

    def test_raise_error_on_wrong_groupby(self, example_dmo_data):
        with pytest.raises(ValueError):
            GenericAggregator(
                **{
                    **GenericAggregator.PredefinedParameters.single_day,
                    "groupby": ["nonexistent"],
                }
            ).aggregate(example_dmo_data)

    def test_nan_considered_true(self, example_dmo_data, dummy_dmo_data_mask):
        mask_with_nan = dummy_dmo_data_mask.copy().replace(True, np.nan)
        agg_with_nan = GenericAggregator(
            **GenericAggregator.PredefinedParameters.single_day
        ).aggregate(example_dmo_data, wb_dmos_mask=mask_with_nan)
        agg_without_nan = GenericAggregator(
            **GenericAggregator.PredefinedParameters.single_day
        ).aggregate(example_dmo_data, wb_dmos_mask=dummy_dmo_data_mask)
        assert_frame_equal(
            agg_with_nan.aggregated_data_, agg_without_nan.aggregated_data_
        )

    def test_no_grouping(self, example_dmo_data, dummy_dmo_data_mask):
        agg = GenericAggregator(
            **(GenericAggregator.PredefinedParameters.single_day | dict(groupby=None))
        ).aggregate(example_dmo_data, wb_dmos_mask=dummy_dmo_data_mask)
        assert len(agg.aggregated_data_) == 1
        assert agg.aggregated_data_.index[0] == "all_wbs"
