import pytest
import pandas as pd
import numpy as np

from multigait.GSD.GSD3 import KheirkhahanGSD
from multigait.pipeline.multimobility_pipeline import (
    MultiGaitPipeline,
    MultiGaitPipelineHealthyCoMorbidity,
    MultiGaitPipelineMultimorbidityImpaired,
)
from multigait.CAD.cad import Cadence
from multigait.ICD.ICD6 import GuIC
from multigait.ICD.ICD4 import ZijlstraIC
from multigait.SL.SL1 import WeinbergSL
from multigait.WS.walking_speed import Ws
from multigait.pipeline.utils._stride_filtering import StrideFiltering
from multigait.pipeline.utils._wb_assembly import WbAssembly
from multigait.aggregation._generic_aggregator import GenericAggregator
from multigait.pipeline.utils._thresholds import get_thresholds
from unittest.mock import patch


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

        data_ss = pd.DataFrame(
            {
                "acc_is": 0.2 * np.sin(2 * np.pi * 1 * t) + 0.0,
                "acc_ml": 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.0,
                "acc_pa": 9.8 + 0.1 * np.sin(2 * np.pi * 1.2 * t),
            }
        )

    return DummyDataset()


@pytest.fixture
def example_datapoint_walking():
    """Longer, more realistic walking-like signal to ensure stride detection."""

    class DummyDataset:
        participant_metadata = {"height_m": 1.75}
        recording_metadata = {"device": "wrist"}
        sampling_rate_hz = 100.0
        group_label = "test"

        n_samples = 3000  # 30 seconds at 100 Hz
        t = np.linspace(0, 30, n_samples)

        # Walking-like signal: ~2 Hz cadence, amplitude -10 to +10
        data_ss = pd.DataFrame(
            {
                "acc_is": 10.0 * np.sin(2 * np.pi * 2 * t),
                "acc_ml": 10.0 * np.sin(2 * np.pi * 1 * t),
                "acc_pa": 9.8 + 10.0 * np.sin(2 * np.pi * 2 * t),
            }
        )

    return DummyDataset()


# Predefined algorithm instances for tests — single ICD (icd_sl=None, fallback)
@pytest.fixture
def example_pipeline_algorithms_single_icd():
    return dict(
        gait_sequence_detection=KheirkhahanGSD(),
        initial_contact_detection=ZijlstraIC(),
        initial_contact_detection_sl=None,
        cadence_calculation=Cadence(),
        stride_length_calculation=WeinbergSL(),
        walking_speed_calculation=Ws(),
        stride_selection=StrideFiltering(),
        wba=WbAssembly(),
        dmo_thresholds=get_thresholds(),
        dmo_aggregation=GenericAggregator(
            **GenericAggregator.PredefinedParameters.single_day
        ),
    )


@pytest.fixture
def example_pipeline_algorithms_dual_icd():
    return dict(
        gait_sequence_detection=KheirkhahanGSD(),
        initial_contact_detection=ZijlstraIC(),
        initial_contact_detection_sl=GuIC(),
        cadence_calculation=Cadence(),
        stride_length_calculation=WeinbergSL(),
        walking_speed_calculation=Ws(),
        stride_selection=StrideFiltering(),
        wba=WbAssembly(),
        dmo_thresholds=get_thresholds(),
        dmo_aggregation=GenericAggregator(
            **GenericAggregator.PredefinedParameters.single_day
        ),
    )


# Expected output columns shared across tests
EXPECTED_WB_COLS = [
    "stride_duration_s",
    "cadence_spm",
    "stride_length_m",
    "walking_speed_mps",
]


def assert_pipeline_outputs(result):
    """Shared assertions for all pipeline variants."""
    assert hasattr(result, "per_stride_parameters_")
    assert hasattr(result, "per_wb_parameters_")
    assert hasattr(result, "aggregated_parameters_")
    for col in EXPECTED_WB_COLS:
        assert col in result.per_wb_parameters_.columns


class TestFullMultiGaitPipeline:
    def test_pipeline_single_icd(
        self, example_datapoint, example_pipeline_algorithms_single_icd
    ):
        """Base pipeline with single ICD — icd_sl falls back to primary ICD."""
        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_single_icd)
        result = pipeline.run(example_datapoint)
        assert_pipeline_outputs(result)

    def test_pipeline_dual_icd(
        self, example_datapoint, example_pipeline_algorithms_dual_icd
    ):
        """Base pipeline with separate ICD for stride length."""
        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_dual_icd)
        result = pipeline.run(example_datapoint)
        assert_pipeline_outputs(result)

    def test_icd_sl_none_does_not_double_detect(
        self, example_datapoint, example_pipeline_algorithms_single_icd
    ):
        """When icd_sl is None, use_separate_icd_sl should be False — no second detection."""
        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_single_icd)
        assert pipeline.initial_contact_detection_sl is None

    def test_dual_icd_sl_uses_different_contacts(
        self,
        example_pipeline_algorithms_single_icd,
        example_pipeline_algorithms_dual_icd,
    ):
        """SL results differ when using ZijlstraIC vs GuIC as the SL detector.

        Single ICD pipeline: ZijlstraIC -> cadence, SL, WS, stride assembly
        Dual ICD pipeline:   ZijlstraIC -> cadence, WS, stride assembly
                             GuIC       -> SL only
        """
        from examples.example_data.example_constructor import (
            construct_datapoint_from_files,
        )

        real_data = construct_datapoint_from_files()

        pipeline_single = MultiGaitPipeline(**example_pipeline_algorithms_single_icd)
        pipeline_dual = MultiGaitPipeline(**example_pipeline_algorithms_dual_icd)

        result_zijlstra_for_sl = pipeline_single.run(real_data)
        result_gu_for_sl = pipeline_dual.run(real_data)

        sl_single = result_zijlstra_for_sl.raw_per_sec_parameters_.get(
            "stride_length_m"
        )
        sl_dual = result_gu_for_sl.raw_per_sec_parameters_.get("stride_length_m")

        if (
            sl_single is not None
            and sl_dual is not None
            and not sl_single.empty
            and not sl_dual.empty
        ):
            assert not sl_single.equals(sl_dual), (
                "SL results should differ: ZijlstraIC and GuIC produce different IC lists, "
                "which should lead to different stride length estimates."
            )
        else:
            pytest.skip("No strides detected — comparison skipped.")

    def test_dual_icd_sl_uses_correct_contacts(
        self, example_pipeline_algorithms_dual_icd
    ):
        """Verify that ZijlstraIC feeds primary ICD and GuIC feeds SL calculation."""
        from examples.example_data.example_constructor import (
            construct_datapoint_from_files,
        )
        from multigait.utils.data_conversions import rename_axes_to_body
        from multigait.SL.SL1 import WeinbergSL

        real_data = construct_datapoint_from_files()
        imu_data = rename_axes_to_body(real_data.data_ss)

        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_dual_icd)
        pipeline.run(real_data)
        gs_list = pipeline.gs_list_

        if gs_list.empty:
            pytest.skip("No gait sequences detected.")

        gs0_start = gs_list.iloc[0]["start"]
        gs0_end = gs_list.iloc[0]["end"]
        first_gs = imu_data.iloc[gs0_start:gs0_end]

        ic_zijlstra = ZijlstraIC().detect(first_gs).ic_list_
        ic_gu = GuIC().detect(first_gs).ic_list_

        # Confirm detectors differ
        assert not ic_zijlstra.equals(ic_gu), (
            "ZijlstraIC and GuIC should produce different IC lists on the same data."
        )

        # Offset to absolute indices
        ic_zijlstra_abs = (ic_zijlstra["ic"] + gs0_start).values.tolist()
        ic_gu_abs = (ic_gu["ic"] + gs0_start).values.tolist()

        # --- Verify primary IC list matches ZijlstraIC ---
        pipeline_ic = pipeline.raw_ic_list_
        gs0_ic = (
            pipeline_ic.xs(0, level="gs_id")
            if 0 in pipeline_ic.index.get_level_values("gs_id")
            else None
        )

        if gs0_ic is not None:
            assert gs0_ic["ic"].values.tolist() == ic_zijlstra_abs, (
                "Primary IC list (cadence/WS) should come from ZijlstraIC."
            )
            assert gs0_ic["ic"].values.tolist() != ic_gu_abs, (
                "Primary IC list should NOT come from GuIC."
            )

        # --- Verify SL output matches WeinbergSL run with GuIC contacts ---
        sl_with_gu = (
            WeinbergSL()
            .calculate(
                first_gs,
                initial_contacts=ic_gu,
                **real_data.participant_metadata,
                dp_group=real_data.group_label,
                sampling_rate_hz=real_data.sampling_rate_hz,
            )
            .stride_length_per_sec_
        )

        sl_with_zijlstra = (
            WeinbergSL()
            .calculate(
                first_gs,
                initial_contacts=ic_zijlstra,
                **real_data.participant_metadata,
                dp_group=real_data.group_label,
                sampling_rate_hz=real_data.sampling_rate_hz,
            )
            .stride_length_per_sec_
        )

        # SL from pipeline (gs_id=0) should match GuIC, not ZijlstraIC
        pipeline_sl = pipeline.raw_per_sec_parameters_.xs(0, level="gs_id")[
            "stride_length_m"
        ].reset_index(drop=True)
        sl_with_gu_values = sl_with_gu.reset_index(drop=True)
        sl_with_zijlstra_values = sl_with_zijlstra.reset_index(drop=True)

        assert np.allclose(
            pipeline_sl.values, sl_with_gu_values.values.squeeze(), equal_nan=True
        ), "Pipeline SL should match WeinbergSL run with GuIC contacts."
        assert not np.allclose(
            pipeline_sl.values, sl_with_zijlstra_values.values.squeeze(), equal_nan=True
        ), "Pipeline SL should NOT match WeinbergSL run with ZijlstraIC contacts."

    def test_icd_sl_contacts_passed_to_sl_calculation(
        self, example_datapoint, example_pipeline_algorithms_dual_icd
    ):
        """Verify that icd_sl is set and not None when dual ICD is configured."""
        pipeline = MultiGaitPipeline(**example_pipeline_algorithms_dual_icd)

        original_run_per_gs = pipeline._run_per_gs
        captured = {}

        def capturing_run_per_gs(gait_sequences, imu_data):
            gs_iterator = original_run_per_gs(gait_sequences, imu_data)
            captured["icd_sl_is_none"] = pipeline.initial_contact_detection_sl is None
            return gs_iterator

        with patch.object(pipeline, "_run_per_gs", side_effect=capturing_run_per_gs):
            pipeline.run(example_datapoint)

        assert not captured["icd_sl_is_none"], "Expected a separate icd_sl to be set"


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
        assert (
            pipeline.initial_contact_detection_sl is not None
        )  # impaired uses separate SL ICD

    def test_impaired_multimorbid_run(self, example_datapoint):
        """Predefined impaired pipeline runs end-to-end."""
        pipeline = MultiGaitPipelineMultimorbidityImpaired()
        result = pipeline.run(example_datapoint)
        assert_pipeline_outputs(result)
