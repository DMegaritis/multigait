import warnings

import numpy as np
import pandas as pd
import pytest

from multigait.utils.interp import map_seconds_to_regions

FS = 100  # sampling rate (Hz); second k covers samples [k * FS, (k + 1) * FS)
N_SEC = 10


def _second_params(values, columns=("a",)):
    """Per-second table indexed like the pipeline: (gs_id, sec_center_samples)."""
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    centers = (np.arange(len(values)) + 0.5) * FS
    index = pd.MultiIndex.from_arrays(
        [np.zeros(len(values), dtype=int), centers.astype(int)],
        names=["gs_id", "sec_center_samples"],
    )
    return pd.DataFrame(values, index=index, columns=list(columns))


def _regions(bounds):
    return pd.DataFrame(bounds, columns=["start", "end"])


def _reference(regions, second_params):
    """Time-weighted mean of the piecewise-constant per-second signal.

    NaN seconds are ignored (mean over the valid part of the region);
    a region without any valid second gives NaN.
    """
    values = second_params.to_numpy()
    out = np.full((len(regions), values.shape[1]), np.nan)
    sec_start = np.arange(len(values)) * FS
    sec_end = sec_start + FS
    for i, (s, e) in enumerate(regions[["start", "end"]].to_numpy()):
        overlap = np.clip(np.minimum(e, sec_end) - np.maximum(s, sec_start), 0, None)
        for j in range(values.shape[1]):
            valid = ~np.isnan(values[:, j])
            if overlap[valid].sum() > 0:
                out[i, j] = (overlap[valid] * values[valid, j]).sum() / overlap[valid].sum()
    return out


def _run(bounds, values, columns=("a",)):
    sec = _second_params(values, columns)
    result = map_seconds_to_regions(_regions(bounds), sec, sampling_rate_hz=FS)
    return result, sec


class TestMapSecondsToRegionsCurrentBehaviour:
    """Must pass before and after the NaN change."""

    @pytest.mark.parametrize("bounds", [[[0, 100]], [[30, 170]], [[250, 730]], [[0, 1000]]])
    def test_constant_signal_gives_constant_mean(self, bounds):
        result, _ = _run(bounds, np.full(N_SEC, 3.5))
        assert np.allclose(result["a"], 3.5)

    def test_region_on_one_second_equals_its_value(self):
        values = np.arange(1.0, N_SEC + 1)
        result, _ = _run([[200, 300], [500, 600]], values)
        assert np.allclose(result["a"], [3.0, 6.0])

    def test_region_spanning_two_seconds_is_time_weighted(self):
        # 30 samples in second 0 (value 1), 70 samples in second 1 (value 2)
        result, _ = _run([[70, 170]], np.arange(1.0, N_SEC + 1))
        assert np.isclose(result["a"].iloc[0], (30 * 1 + 70 * 2) / 100)

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_reference_without_nan(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.uniform(0.5, 2.0, size=(N_SEC, 2))
        starts = rng.integers(0, 800, size=8)
        bounds = np.column_stack([starts, starts + rng.integers(20, 200, size=8)])
        result, sec = _run(bounds, values, columns=("a", "b"))
        assert np.allclose(result[["a", "b"]].to_numpy(), _reference(_regions(bounds), sec))

    def test_keeps_region_columns_and_index(self):
        regions = _regions([[0, 100], [100, 250]])
        regions["extra"] = [7, 8]
        regions.index = pd.Index([10, 11], name="s_id")
        sec = _second_params(np.ones(N_SEC), columns=("a",))
        result = map_seconds_to_regions(regions, sec, sampling_rate_hz=FS)
        assert list(result.columns) == ["start", "end", "extra", "a"]
        assert result.index.equals(regions.index)
        assert result["extra"].tolist() == [7, 8]

    def test_empty_regions_adds_columns(self):
        sec = _second_params(np.ones(N_SEC))
        result = map_seconds_to_regions(_regions(np.empty((0, 2))), sec, sampling_rate_hz=FS)
        assert result.empty
        assert "a" in result.columns

    def test_empty_second_params_gives_nan_columns(self):
        sec = _second_params(np.ones(N_SEC)).iloc[0:0]
        result = map_seconds_to_regions(_regions([[0, 100]]), sec, sampling_rate_hz=FS)
        assert result["a"].isna().all()

    def test_non_float_column_raises(self):
        sec = _second_params(np.ones(N_SEC))
        sec["a"] = sec["a"].astype(int)
        with pytest.raises(ValueError, match="Non-float"):
            map_seconds_to_regions(_regions([[0, 100]]), sec, sampling_rate_hz=FS)


class TestMapSecondsToRegionsNaNHandling:
    """Desired behaviour: NaN seconds are ignored, they must not blank other regions."""

    REGIONS = [[0, 100], [100, 200], [250, 350], [450, 550], [650, 750], [850, 950]]

    def test_leading_nan_does_not_blank_the_sequence(self):
        values = np.ones(N_SEC)
        values[0] = np.nan
        result, _ = _run(self.REGIONS, values)
        assert np.isnan(result["a"].iloc[0])  # region fully inside the NaN second
        assert np.allclose(result["a"].iloc[1:], 1.0)

    def test_middle_nan_only_affects_overlapping_regions(self):
        values = np.ones(N_SEC)
        values[4] = np.nan  # samples 400-500
        result, _ = _run(self.REGIONS, values)
        assert np.allclose(result["a"].iloc[:3], 1.0)  # before the gap
        assert np.allclose(result["a"].iloc[4:], 1.0)  # after the gap (blank today)
        # region 450-550 overlaps the gap by 50 samples: mean over its valid part
        assert np.isclose(result["a"].iloc[3], 1.0)

    def test_trailing_nan_only_affects_last_regions(self):
        values = np.ones(N_SEC)
        values[-2:] = np.nan  # samples 800-1000
        result, _ = _run(self.REGIONS, values)
        assert np.allclose(result["a"].iloc[:-1], 1.0)
        assert np.isnan(result["a"].iloc[-1])

    def test_region_straddling_a_nan_second_uses_valid_part_only(self):
        values = np.array([1.0, 2.0, np.nan, 4.0] + [1.0] * (N_SEC - 4))
        result, _ = _run([[150, 350]], values)  # 50 in s1 (2.0), 100 in s2 (NaN), 50 in s3 (4.0)
        assert np.isclose(result["a"].iloc[0], (50 * 2.0 + 50 * 4.0) / 100)

    def test_all_nan_gives_nan(self):
        result, _ = _run(self.REGIONS, np.full(N_SEC, np.nan))
        assert result["a"].isna().all()

    def test_columns_are_independent(self):
        values = np.ones((N_SEC, 2))
        values[0, 0] = np.nan  # NaN only in column "a"
        result, _ = _run(self.REGIONS, values, columns=("a", "b"))
        assert np.allclose(result["b"], 1.0)
        assert np.allclose(result["a"].iloc[1:], 1.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_reference_with_random_nan(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.uniform(0.5, 2.0, size=(N_SEC, 2))
        values[rng.random(values.shape) < 0.3] = np.nan
        starts = rng.integers(0, 800, size=10)
        bounds = np.column_stack([starts, starts + rng.integers(20, 200, size=10)])
        result, sec = _run(bounds, values, columns=("a", "b"))
        assert np.allclose(
            result[["a", "b"]].to_numpy(),
            _reference(_regions(bounds), sec),
            equal_nan=True,
        )


class TestMapSecondsToRegionsEdgeCases:
    """Additional checks for the NaN-aware implementation."""

    @pytest.mark.parametrize(
        "values",
        [
            [np.nan] + [1.0] * (N_SEC - 1),  # leading NaN
            [1.0] * (N_SEC - 1) + [np.nan],  # trailing NaN
            [np.nan] * N_SEC,  # all NaN
        ],
    )
    def test_no_warnings_with_nan(self, values):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _run([[0, 100], [250, 450], [900, 1000]], values)

    def test_zero_length_region_is_nan_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result, _ = _run([[200, 200]], np.ones(N_SEC))
        assert np.isnan(result["a"].iloc[0])

    def test_long_nan_run_does_not_affect_later_regions(self):
        values = np.array([1.0, 1.0] + [np.nan] * 4 + [2.0] * (N_SEC - 6))
        result, _ = _run([[0, 200], [250, 350], [650, 750], [700, 1000]], values)
        assert np.isclose(result["a"].iloc[0], 1.0)
        assert np.isnan(result["a"].iloc[1])  # fully inside the NaN run
        assert np.isclose(result["a"].iloc[2], 2.0)
        assert np.isclose(result["a"].iloc[3], 2.0)

    def test_alternating_nan_seconds(self):
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0, np.nan, 7.0, np.nan, 9.0, np.nan])
        bounds = [[0, 200], [100, 300], [200, 400], [0, 1000]]
        result, sec = _run(bounds, values)
        # a region over one valid and one NaN second equals the valid value
        assert np.allclose(result["a"].iloc[:3], [1.0, 3.0, 3.0])
        assert np.isclose(result["a"].iloc[3], np.mean([1, 3, 5, 7, 9]))
        assert np.allclose(result[["a"]].to_numpy(), _reference(_regions(bounds), sec))

    def test_single_second_input(self):
        result, _ = _run([[10, 90]], [2.0])
        assert np.isclose(result["a"].iloc[0], 2.0)
        result_nan, _ = _run([[10, 90]], [np.nan])
        assert np.isnan(result_nan["a"].iloc[0])

    def test_region_beyond_the_data_keeps_edge_value(self):
        """Documented behaviour: values continue past the edges (extrapolation)."""
        result, _ = _run([[900, 1100]], np.full(N_SEC, 1.5))
        assert np.isclose(result["a"].iloc[0], 1.5)

    def test_region_beyond_trailing_nan_is_nan(self):
        values = np.ones(N_SEC)
        values[-2:] = np.nan
        result, _ = _run([[900, 1100]], values)
        assert np.isnan(result["a"].iloc[0])

    def test_float32_dtype_is_preserved(self):
        sec = _second_params(np.ones(N_SEC)).astype(np.float32)
        result = map_seconds_to_regions(_regions([[0, 100]]), sec, sampling_rate_hz=FS)
        assert result["a"].dtype == np.float32

    def test_inputs_are_not_modified(self):
        sec = _second_params(np.array([1.0, np.nan, 3.0] + [1.0] * (N_SEC - 3)))
        regions = _regions([[0, 250], [100, 400]])
        sec_before, regions_before = sec.copy(), regions.copy()
        map_seconds_to_regions(regions, sec, sampling_rate_hz=FS)
        pd.testing.assert_frame_equal(sec, sec_before)
        pd.testing.assert_frame_equal(regions, regions_before)

    def test_gait_sequences_do_not_affect_each_other(self):
        """As in the pipeline, the function runs once per gs_id: NaN in one must not leak into another."""
        centers = ((np.arange(N_SEC) + 0.5) * FS).astype(int)
        values_gs0 = np.ones(N_SEC)
        values_gs0[0] = np.nan  # leading NaN in gait sequence 0 only
        values_gs1 = np.full(N_SEC, 2.0)
        sec = pd.DataFrame(
            {"a": np.concatenate([values_gs0, values_gs1])},
            index=pd.MultiIndex.from_arrays(
                [np.repeat([0, 1], N_SEC), np.tile(centers, 2)],
                names=["gs_id", "sec_center_samples"],
            ),
        )
        bounds = [[150, 350], [450, 750]]
        results = {
            gs_id: map_seconds_to_regions(
                _regions(bounds), sec_gs, sampling_rate_hz=FS
            )["a"].to_numpy()
            for gs_id, sec_gs in sec.groupby(level="gs_id")
        }
        assert np.allclose(results[0], 1.0)
        assert np.allclose(results[1], 2.0)

    @pytest.mark.parametrize("seed", range(5))
    def test_many_regions_longer_than_a_second(self, seed):
        rng = np.random.default_rng(100 + seed)
        n_sec = 60
        values = rng.uniform(0.3, 2.5, size=(n_sec, 3))
        values[rng.random(values.shape) < 0.4] = np.nan
        sec = _second_params(values, columns=("a", "b", "c"))
        starts = rng.integers(0, n_sec * FS - 400, size=500)
        bounds = np.column_stack([starts, starts + rng.integers(1, 400, size=500)])
        result = map_seconds_to_regions(_regions(bounds), sec, sampling_rate_hz=FS)
        assert np.allclose(
            result[["a", "b", "c"]].to_numpy(),
            _reference(_regions(bounds), sec),
            equal_nan=True,
        )