import pytest
import pandas as pd
import numpy as np
from multigait.pipeline.multimobility_pipeline import (
    MultiGaitPipeline,
    MultiGaitPipelineHealthyCoMorbidity,
    MultiGaitPipelineMultimorbidityImpaired,
)
from multigait.CAD.cad import Cadence
from multigait.GSD.GSD2 import HickeyGSD
from multigait.ICD.ICD2 import McCamleyIC
from multigait.ICD.ICD4 import ZijlstraIC
from multigait.SL.SL1 import WeinbergSL
from multigait.WS.walking_speed import Ws
from multigait.pipeline.utils._stride_filtering import StrideFiltering
from multigait.pipeline.utils._wb_assembly import WbAssembly
from multigait.aggregation._generic_aggregator import GenericAggregator
from multigait.pipeline.utils._thresholds import get_thresholds


# Minimal example GaitDatasetT fixture
@pytest.fixture
def example_datapoint():
    class DummyDataset:
        participant_metadata = {"height_m": 1.75}
        recording_metadata = {"device": "wrist"}
        sampling_rate_hz = 100.0
        group_label = "test"

        n_samples = 500
        t = np.linspace(0, 1, n_samples)

        data_ss = pd.DataFrame({
            "acc_is": 0.2 * np.sin(2 * np.pi * 1 * t) + 0.0,
            "acc_ml": 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.0,
            "acc_pa": 9.8 + 0.1 * np.sin(2 * np.pi * 1.2 * t)
        })

    return DummyDataset()


# Predefined algorithm instances for tests — single ICD (icd_sl=None, fallback)
@pytest.fixture
def example_pipeline_algorithms_single_icd():
    return dict(
        gait_sequence_detection=HickeyGSD(),
        initial_contact_detection=McCamleyIC(),
        initial_contact_detection_sl=None,  # explicit fallback: no separate SL detector
        cadence_calculation=Cadence(),
        stride_length_calculation=WeinbergSL(),
        walking_speed_calculation=Ws(),
        stride_selection=StrideFiltering(),
        wba=WbAssembly(),
        dmo_thresholds=get_thresholds(),
        dmo_aggregation=GenericAggregator(**GenericAggregator.PredefinedParameters.single_day),
    )


# Predefined algorithm instances for tests — separate ICD for SL
@pytest.fixture
def example_pipeline_algorithms_dual_icd():
    return dict(
        gait_sequence_detection=HickeyGSD(),
        initial_contact_detection=McCamleyIC(),
        initial_contact_detection_sl=ZijlstraIC(),  # separate SL detector
        cadence_calculation=Cadence(),
        stride_length_calculation=WeinbergSL(),
        walking_speed_calculation=Ws(),
        stride_selection=StrideFiltering(),
        wba=WbAssembly(),
        dmo_thresholds=get_thresholds(),
        dmo_aggregation=GenericAggregator(**GenericAggregator.PredefinedParameters.single_day),
    )


# Expected output columns shared across tests
EXPECTED_WB_COLS = [
    "stride_duration_s", "cadence_spm", "stride_length_m", "walking_speed_mps"
]


def assert_pipeline_outputs(result):
    """Shared assertions for all pipeline variants."""
    assert hasattr(result, "per_stride_parameters_")
    assert hasattr(result, "per_wb_parameters_")
    assert hasattr(result, "aggregated_parameters_")
    for col in EXPECTED_WB_COLS:
        assert col in result.per_wb_parameters_.columns


class TestFullMultiGaitPipeline:
    def test_pipeline_single_icd(self, example_datapoint, example_pipeline_algorithms_single_icd):
        """Base pipeline with single ICD — icd_sl falls back to primary ICD."""
        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_single_icd)
        result = pipeline.run(example_datapoint)
        assert_pipeline_outputs(result)

    def test_pipeline_dual_icd(self, example_datapoint, example_pipeline_algorithms_dual_icd):
        """Base pipeline with separate ICD for stride length."""
        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_dual_icd)
        result = pipeline.run(example_datapoint)
        assert_pipeline_outputs(result)

    def test_icd_sl_none_does_not_double_detect(self, example_datapoint, example_pipeline_algorithms_single_icd):
        """When icd_sl is None, use_separate_icd_sl should be False — no second detection."""
        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_single_icd)
        assert pipeline.initial_contact_detection_sl is None


class TestPredefinedPipelines:
    def test_healthy_comorbidity_instantiation(self):
        """Predefined healthy pipeline instantiates with no arguments."""
        pipeline = MultiGaitPipelineHealthyCoMorbidity()
        assert pipeline.initial_contact_detection_sl is None  # healthy uses fallback

    def test_healthy_comorbidity_run(self, example_datapoint):
        """Predefined healthy pipeline runs end-to-end."""
        pipeline = MultiGaitPipelineHealthyCoMorbidity()
        result = pipeline.run(example_datapoint)
        assert_pipeline_outputs(result)

    def test_impaired_multimorbid_instantiation(self):
        """Predefined impaired pipeline instantiates with no arguments."""
        pipeline = MultiGaitPipelineMultimorbidityImpaired()
        assert pipeline.initial_contact_detection_sl is not None  # impaired uses separate SL ICD

    def test_impaired_multimorbid_run(self, example_datapoint):
        """Predefined impaired pipeline runs end-to-end."""
        pipeline = MultiGaitPipelineMultimorbidityImpaired()
        result = pipeline.run(example_datapoint)
        assert_pipeline_outputs(result)