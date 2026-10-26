import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from multigait.pipeline.utils.ic_to_stride import strides_list_from_ic_list_no_lrc

SAMPLING_RATE_HZ = 100.0


def _make_ic_list(ics, gs_id=None):
    """Build an IC list shaped like the pipeline's: index (gs_id, step_id), column "ic"."""
    ics = list(ics)
    if gs_id is None:
        return pd.DataFrame({"ic": ics}, index=pd.RangeIndex(len(ics), name="step_id"))
    index = pd.MultiIndex.from_arrays(
        [[gs_id] * len(ics), range(len(ics))], names=["gs_id", "step_id"]
    )
    return pd.DataFrame({"ic": ics}, index=index)


@pytest.fixture
def alternating_ics():
    """Six contacts with a constant step time of 0.55 s (55 samples at 100 Hz)."""
    return list(range(100, 100 + 6 * 55, 55))


@pytest.fixture
def ic_list_single_gs(alternating_ics):
    return _make_ic_list(alternating_ics, gs_id=0)


class TestStridesListFromIcListNoLrc:
    """Tests for the stride list built from initial contacts without left/right labels."""

    def test_stride_spans_two_contacts(self):
        """A stride goes from IC i to IC i+2 (full gait cycle), not to IC i+1 (a step)."""
        ic_list = _make_ic_list([100, 160, 225, 285, 350, 410])
        result = strides_list_from_ic_list_no_lrc(ic_list)

        assert result["start"].tolist() == [100, 160, 225, 285]
        assert result["end"].tolist() == [225, 285, 350, 410]

    @pytest.mark.parametrize("n_contacts", [3, 4, 5, 6, 7, 10, 25])
    def test_n_strides_is_n_contacts_minus_two(self, n_contacts):
        ic_list = _make_ic_list(np.arange(n_contacts) * 55 + 100)
        result = strides_list_from_ic_list_no_lrc(ic_list)
        assert len(result) == n_contacts - 2

    def test_four_strides_need_six_contacts(self):
        """Matches the WB definition (min 4 strides): R-L-R-L-R-L is the minimum, 5 contacts are not enough."""
        five = strides_list_from_ic_list_no_lrc(_make_ic_list(range(0, 5 * 55, 55)))
        six = strides_list_from_ic_list_no_lrc(_make_ic_list(range(0, 6 * 55, 55)))
        assert len(five) == 3
        assert len(six) == 4

    @pytest.mark.parametrize("n_contacts", [1, 2])
    def test_too_few_contacts_returns_empty(self, n_contacts):
        ic_list = _make_ic_list(np.arange(n_contacts) * 55 + 100)
        result = strides_list_from_ic_list_no_lrc(ic_list)
        assert result.empty
        assert list(result.columns) == ["start", "end"]

    def test_three_contacts_give_one_stride(self):
        result = strides_list_from_ic_list_no_lrc(_make_ic_list([100, 160, 225]))
        assert result["start"].tolist() == [100]
        assert result["end"].tolist() == [225]

    def test_empty_input(self):
        empty = pd.DataFrame(
            {"ic": pd.Series([], dtype="int64")},
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["gs_id", "step_id"]
            ),
        )
        result = strides_list_from_ic_list_no_lrc(empty)
        assert result.empty
        assert list(result.columns) == ["start", "end"]
        assert result.index.names[-1] == "s_id"

    def test_output_columns_dtypes_and_index_name(self, ic_list_single_gs):
        result = strides_list_from_ic_list_no_lrc(ic_list_single_gs)
        assert list(result.columns) == ["start", "end"]
        assert result["start"].dtype == np.int64
        assert result["end"].dtype == np.int64
        # last index level is renamed to s_id, outer level is preserved
        assert result.index.names == ["gs_id", "s_id"]

    def test_index_follows_starting_contact(self, ic_list_single_gs):
        """Each stride keeps the index label of the contact it starts at."""
        result = strides_list_from_ic_list_no_lrc(ic_list_single_gs)
        assert result.index.tolist() == ic_list_single_gs.index[:-2].tolist()

    def test_unsorted_input_is_sorted_by_contact_time(self, alternating_ics):
        rng = np.random.default_rng(0)
        shuffled = rng.permutation(alternating_ics)
        result_sorted = strides_list_from_ic_list_no_lrc(_make_ic_list(alternating_ics))
        result_shuffled = strides_list_from_ic_list_no_lrc(_make_ic_list(shuffled))
        assert result_shuffled["start"].tolist() == result_sorted["start"].tolist()
        assert result_shuffled["end"].tolist() == result_sorted["end"].tolist()

    def test_strides_overlap_and_are_ordered_by_start(self, ic_list_single_gs):
        """Consecutive strides overlap by one step, as left/right strides do in mobgap."""
        result = strides_list_from_ic_list_no_lrc(ic_list_single_gs)
        assert result["start"].is_monotonic_increasing
        assert (result["end"] > result["start"]).all()
        # stride i+1 starts before stride i ends
        assert (result["start"].iloc[1:].to_numpy() < result["end"].iloc[:-1].to_numpy()).all()

    def test_stride_duration_is_twice_step_duration(self):
        """Regression test: the old version returned step time as 'stride duration'.

        With a constant step time of 0.55 s the stride duration must be 1.10 s and
        120 / stride-based cadence must agree with it.
        """
        step_time_s = 0.55
        ics = (np.arange(8) * step_time_s * SAMPLING_RATE_HZ + 100).astype(int)
        result = strides_list_from_ic_list_no_lrc(_make_ic_list(ics))
        stride_duration_s = (result["end"] - result["start"]) / SAMPLING_RATE_HZ

        assert np.allclose(stride_duration_s, 2 * step_time_s)
        cadence_spm = 60.0 / step_time_s
        assert np.allclose(120.0 / cadence_spm, stride_duration_s)

    def test_bout_extent_is_unchanged(self, alternating_ics):
        """First stride starts at the first contact, last stride ends at the last contact."""
        result = strides_list_from_ic_list_no_lrc(_make_ic_list(alternating_ics))
        assert result["start"].min() == alternating_ics[0]
        assert result["end"].max() == alternating_ics[-1]

    def test_input_not_modified(self, ic_list_single_gs):
        original = ic_list_single_gs.copy()
        strides_list_from_ic_list_no_lrc(ic_list_single_gs)
        assert_frame_equal(ic_list_single_gs, original)

    def test_groupby_does_not_cross_gait_sequences(self):
        """As used in the pipeline: applied per gs_id, strides never span two gait sequences."""
        gs0 = [100, 155, 210, 265, 320, 375]  # 6 contacts -> 4 strides
        gs1 = [5000, 5060, 5120, 5180, 5240]  # 5 contacts -> 3 strides
        ic_list = pd.concat([_make_ic_list(gs0, gs_id=0), _make_ic_list(gs1, gs_id=1)])

        stride_list = ic_list.groupby("gs_id", group_keys=False).apply(
            strides_list_from_ic_list_no_lrc
        )

        assert len(stride_list) == (len(gs0) - 2) + (len(gs1) - 2)
        gs_ids = stride_list.index.get_level_values("gs_id")
        for gs_id, contacts in ((0, gs0), (1, gs1)):
            strides = stride_list[gs_ids == gs_id]
            assert strides["start"].isin(contacts).all()
            assert strides["end"].isin(contacts).all()
            assert strides["end"].max() <= max(contacts)
            assert strides["start"].min() >= min(contacts)


class TestPipelineStrideOutputs:
    """Pipeline-level checks that need the example data (same data as test_pipeline.py)."""

    @pytest.fixture
    def pipeline_result(self):
        from examples.example_data.example_constructor import (
            construct_datapoint_from_files,
        )
        from multigait.pipeline import MultiGaitPipelineHealthyCoMorbidity

        result = MultiGaitPipelineHealthyCoMorbidity().run(
            construct_datapoint_from_files()
        )
        if result.per_wb_parameters_.empty:
            pytest.skip("No walking bouts detected.")
        return result

    def test_min_contacts_per_wb(self, pipeline_result):
        """min_strides=4 means every WB has at least 6 contacts."""
        wb = pipeline_result.per_wb_parameters_
        assert (wb["n_strides"] >= 4).all()
        assert (wb["n_raw_initial_contacts"] >= 6).all()

    def test_n_strides_not_more_than_contacts_minus_two(self, pipeline_result):
        """Strides dropped by the stride filter can only reduce the count."""
        wb = pipeline_result.per_wb_parameters_
        assert (wb["n_strides"] <= wb["n_raw_initial_contacts"] - 2).all()

    def test_stride_duration_consistent_with_cadence(self, pipeline_result):
        """Stride duration is two steps: 120 / cadence should be close to it."""
        wb = pipeline_result.per_wb_parameters_.dropna(
            subset=["stride_duration_s", "cadence_spm"]
        )
        if wb.empty:
            pytest.skip("No walking bouts with both cadence and stride duration.")
        ratio = (120.0 / wb["cadence_spm"]) / wb["stride_duration_s"]
        assert 0.8 < ratio.median() < 1.2