import typing
import warnings
from types import MappingProxyType
from typing import Final
import numpy as np
import pandas as pd
from pandas import option_context
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack
from multigait.aggregation import AggregatorBase


def _custom_quantile(x: pd.Series) -> float:
    """Calculate the 90th percentile of the passed data."""
    if x.isna().all():
        return np.nan
    return np.nanpercentile(x, 90)


def _coefficient_of_variation(x: pd.Series) -> float:
    """Calculate variation of the passed data."""
    return x.std() / x.mean()


class GenericAggregator(AggregatorBase):
    """
    Aggregation algorithm used in the Multimobility framework.

    This estimator aggregates digital mobility outcomes (DMOs) from individual
    walking bouts into a set of higher-level summary parameters. Aggregation is
    performed across predefined walking-bout–duration ranges as well as for the
    complete, unfiltered set of bouts. For each duration group, multiple scalar
    summary statistics are calculated (for example mean, median, count, 90th
    percentile, or coefficient of variation).

    Aggregation logic
    -----------------
    Aggregations are defined through a set of filters and parameter mappings that
    specify which DMOs are summarized and how. The following duration categories
    are implemented:

    - No duration filter: key suffix "_all"
    - 10 < duration_s ≤ 30: key contains "1030"
    - duration_s > 10: key contains "10"
    - duration_s > 30: key contains "30"
    - duration_s > 60: key contains "60"

    For each category a defined set of aggregate parameters is produced based on
    the columns available in the input data. If a column required for a parameter
    is not present, that parameter is skipped.

    Supported metrics
    -----------------
    The estimator supports aggregating both per-bout values and bout-level
    variability metrics. Each aggregated parameter follows a naming convention:

    - Percentile:    ``*_p90``                 (representing max)
    - Coefficient of variation across bouts: `*_var` (computed from per-bout aggregated values)
    - Within-bout coefficient of variation: `*_wb_cv` (computed from variability inside each individual bout)
    - Within-bout RMSSD: ``*_wb_rmssd`` (computed from variability inside each individual bout)

    The columns required to compute all available parameters are defined in
    ``INPUT_COLUMNS``. If a corresponding per-bout CV or RMSSD column is present,
    its aggregated mean is included in the output.

    Masking and data validity
    -------------------------
    The optional ``wb_dmos_mask`` parameter allows exclusion of implausible values
    at the element level. The mask must align exactly with the input index after
    grouping and uses the following semantics:

    - False in "duration_s": the entire bout is removed
    - False in "walking_speed_mps": only that value is excluded
    - False in "stride_length_m": excludes both stride length and walking speed
    - False in "cadence_spm": excludes both cadence and walking speed
    - False in "stride_duration_s": excludes stride duration

    Mask values of NaN are interpreted as True for backwards compatibility.

    Grouping behavior
    -----------------
    Aggregations are calculated per group defined by ``groupby``. Typical use cases
    include:

    - A single summary across all bouts in the dataset: ``groupby=None``
    - Daily summaries for each participant:
      ``groupby=["participant_id", "measurement_date"]``

    The ``unique_wb_id_column`` must uniquely identify bouts within each group.

    Parameter naming
    ----------------
    Aggregated parameters use descriptive column names by default. If
    ``use_original_names=True``, names are mapped using the ``ALTERNATIVE_NAMES``
    dictionary to the historical naming scheme.

    Attributes
    ----------
    aggregated_data_ : pandas.DataFrame
        The aggregated DMO results, indexed by grouping variables.
    filtered_wb_dmos_ : pandas.DataFrame
        A filtered version of the input data reflecting mask-based exclusion.

    Notes
    -----
    The available output parameters are determined dynamically based on the
    presence of corresponding per-bout input columns. When extending the
    implementation with new parameters, ``ALTERNATIVE_NAMES`` should be updated
    if the original naming scheme should remain supported.
    """

    ALTERNATIVE_NAMES: typing.ClassVar[dict[str, str]] = {
        "wb_all_sum": "wb_all__count",
        "walkdur_all_sum": "total_walking_duration_min",
        "wbdur_all_avg": "wb_all__duration_s__avg",
        "wbdur_all_p90": "wb_all__duration_s__p90",
        "wbdur_all_var": "wb_all__duration_s__var",
        "cadence_all_avg": "wb_all__cadence_spm__avg",
        "strdur_all_avg": "wb_all__stride_duration_s__avg",
        "cadence_all_var": "wb_all__cadence_spm__var",
        "strdur_all_var": "wb_all__stride_duration_s__var",
        "cadence_all_wb_cv": "wb_all__cadence_spm__cv__avg",
        "strdur_all_wb_cv": "wb_all__stride_duration_s__cv__avg",
        "cadence_all_wb_rmssd": "wb_all__cadence_spm__rmssd__avg",
        "strdur_all_wb_rmssd": "wb_all__stride_duration_s__rmssd__avg",
        "wb_1030_sum": "wb_10_30__count",
        "ws_1030_avg": "wb_10_30__walking_speed_mps__avg",
        "strlen_1030_avg": "wb_10_30__stride_length_m__avg",
        "ws_1030_wb_cv": "wb_10_30__walking_speed_mps__cv__avg",
        "strlen_1030_wb_cv": "wb_10_30__stride_length_m__cv__avg",
        "ws_1030_wb_rmssd": "wb_10_30__walking_speed_mps__rmssd__avg",
        "strlen_1030_wb_rmssd": "wb_10_30__stride_length_m__rmssd__avg",
        "wb_10_sum": "wb_10__count",
        "ws_10_p90": "wb_10__walking_speed_mps__p90",
        "wb_30_sum": "wb_30__count",
        "ws_30_avg": "wb_30__walking_speed_mps__avg",
        "strlen_30_avg": "wb_30__stride_length_m__avg",
        "cadence_30_avg": "wb_30__cadence_spm__avg",
        "strdur_30_avg": "wb_30__stride_duration_s__avg",
        "ws_30_p90": "wb_30__walking_speed_mps__p90",
        "cadence_30_p90": "wb_30__cadence_spm__p90",
        "ws_30_var": "wb_30__walking_speed_mps__var",
        "strlen_30_var": "wb_30__stride_length_m__var",
        "cadence_30_wb_cv": "wb_30__cadence_spm__cv__avg",
        "strdur_30_wb_cv": "wb_30__stride_duration_s__cv__avg",
        "ws_30_wb_cv": "wb_30__walking_speed_mps__cv__avg",
        "strlen_30_wb_cv": "wb_30__stride_length_m__cv__avg",
        "cadence_30_wb_rmssd": "wb_30__cadence_spm__rmssd__avg",
        "strdur_30_wb_rmssd": "wb_30__stride_duration_s__rmssd__avg",
        "ws_30_wb_rmssd": "wb_30__walking_speed_mps__rmssd__avg",
        "strlen_30_wb_rmssd": "wb_30__stride_length_m__rmssd__avg",
        "wb_60_sum": "wb_60__count",
        "wbsteps_all_sum": "wb_all__n_raw_initial_contacts__sum",
        "alpha": "alpha",
    }

    INPUT_COLUMNS: typing.ClassVar[list[str]] = [
        "stride_duration_s",
        "stride_duration_s_cv",
        "stride_duration_s_rmssd",
        "n_raw_initial_contacts",
        "walking_speed_mps",
        "walking_speed_mps_cv",
        "walking_speed_mps_rmssd",
        "stride_length_m",
        "stride_length_m_cv",
        "stride_length_m_rmssd",
        "cadence_spm",
        "cadence_spm_cv",
        "cadence_spm_rmssd",
        "alpha",
    ]

    _ALL_WB_AGGS: typing.ClassVar[
        dict[str, tuple[str, typing.Union[str, typing.Callable]]]
    ] = {
        "wb_all_sum": ("duration_s", "count"),
        "walkdur_all_sum": ("duration_s", "sum"),
        "wbsteps_all_sum": ("n_raw_initial_contacts", "sum"),
        "wbdur_all_avg": ("duration_s", "median"),
        "wbdur_all_p90": ("duration_s", _custom_quantile),
        "wbdur_all_var": ("duration_s", _coefficient_of_variation),
        "cadence_all_avg": ("cadence_spm", "mean"),
        "strdur_all_avg": ("stride_duration_s", "mean"),
        "cadence_all_var": ("cadence_spm", _coefficient_of_variation),
        "strdur_all_var": ("stride_duration_s", _coefficient_of_variation),
        "cadence_all_wb_cv": ("cadence_spm_cv", "mean"),
        "strdur_all_wb_cv": ("stride_duration_s_cv", "mean"),
        "cadence_all_wb_rmssd": ("cadence_spm_rmssd", "mean"),
        "strdur_all_wb_rmssd": ("stride_duration_s_rmssd", "mean"),
        "alpha": ("alpha", "mean"),
    }

    _TEN_THIRTY_WB_AGGS: typing.ClassVar = {
        "wb_1030_sum": ("duration_s", "count"),
        "ws_1030_avg": ("walking_speed_mps", "mean"),
        "strlen_1030_avg": ("stride_length_m", "mean"),
        "ws_1030_wb_cv": ("walking_speed_mps_cv", "mean"),
        "strlen_1030_wb_cv": ("stride_length_m_cv", "mean"),
        "ws_1030_wb_rmssd": ("walking_speed_mps_rmssd", "mean"),
        "strlen_1030_wb_rmssd": ("stride_length_m_rmssd", "mean"),
    }

    _TEN_WB_AGGS: typing.ClassVar = {
        "wb_10_sum": ("duration_s", "count"),
        "ws_10_p90": ("walking_speed_mps", _custom_quantile),
    }

    _THIRTY_WB_AGGS: typing.ClassVar = {
        "wb_30_sum": ("duration_s", "count"),
        "ws_30_avg": ("walking_speed_mps", "mean"),
        "strlen_30_avg": ("stride_length_m", "mean"),
        "cadence_30_avg": ("cadence_spm", "mean"),
        "strdur_30_avg": ("stride_duration_s", "mean"),
        "ws_30_p90": ("walking_speed_mps", _custom_quantile),
        "cadence_30_p90": ("cadence_spm", _custom_quantile),
        "ws_30_var": ("walking_speed_mps", _coefficient_of_variation),
        "strlen_30_var": ("stride_length_m", _coefficient_of_variation),
        "cadence_30_wb_cv": ("cadence_spm_cv", "mean"),
        "strdur_30_wb_cv": ("stride_duration_s_cv", "mean"),
        "ws_30_wb_cv": ("walking_speed_mps_cv", "mean"),
        "strlen_30_wb_cv": ("stride_length_m_cv", "mean"),
        "cadence_30_wb_rmssd": ("cadence_spm_rmssd", "mean"),
        "strdur_30_wb_rmssd": ("stride_duration_s_rmssd", "mean"),
        "ws_30_wb_rmssd": ("walking_speed_mps_rmssd", "mean"),
        "strlen_30_wb_rmssd": ("stride_length_m_rmssd", "mean"),
    }

    _SIXTY_WB_AGGS: typing.ClassVar = {"wb_60_sum": ("duration_s", "count")}

    _FILTERS_AND_AGGS: typing.ClassVar = [
        (None, _ALL_WB_AGGS),
        ("duration_s > 10 & duration_s <= 30", _TEN_THIRTY_WB_AGGS),
        ("duration_s > 10", _TEN_WB_AGGS),
        ("duration_s > 30", _THIRTY_WB_AGGS),
        ("duration_s > 60", _SIXTY_WB_AGGS),
    ]

    _UNIT_CONVERSIONS: typing.ClassVar = {
        "walkdur_all_sum": 1 / 60,
    }

    _COUNT_COLUMNS: typing.ClassVar = [
        "wb_10_sum",
        "wb_30_sum",
        "wb_60_sum",
        "wb_all_sum",
        "steps_all_sum",
    ]

    groupby: typing.Optional[typing.Sequence[str]]
    unique_wb_id_column: str

    wb_dmos_mask: pd.DataFrame

    filtered_wb_dmos_: pd.DataFrame

    class PredefinedParameters:
        multimobility_data: Final = MappingProxyType(
            {
                "groupby": ["participant_id", "measurement_date"],
                "unique_wb_id_column": "wb_id",
                "use_original_names": True,
            }
        )

        multimobility_data_date: Final = MappingProxyType(
            {
                "groupby": ["measurement_date"],
                "unique_wb_id_column": "wb_id",
                "use_original_names": True,
            }
        )

        single_day: Final = MappingProxyType(
            {
                "groupby": None,
                "unique_wb_id_column": "wb_id",
                "use_original_names": False,
            }
        )

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.single_day.items()})
    def __init__(
        self,
        groupby: typing.Optional[typing.Sequence[str]],
        *,
        unique_wb_id_column: str,
        use_original_names: bool,
    ) -> None:
        self.groupby = groupby
        self.unique_wb_id_column = unique_wb_id_column
        self.use_original_names = use_original_names

    def aggregate(  # noqa: C901
        self,
        wb_dmos: pd.DataFrame,
        *,
        wb_dmos_mask: typing.Union[pd.DataFrame, None] = None,
        **_: Unpack[dict[str, typing.Any]],
    ) -> Self:
        """%(aggregate_short)s.

        Parameters
        ----------
        %(aggregate_para)s
        wb_dmos_mask
            A boolean DataFrame with the same shape the ``wb_dmos`` indicating the validity of every measure.
            If the DataFrame contains a ``NaN`` value, this is interpreted as ``True``, assuming no checks were applied
            to this value and the corresponding measure is regarded as plausible.

        %(aggregate_return)s
        """
        self.wb_dmos = wb_dmos
        self.wb_dmos_mask = wb_dmos_mask
        groupby = self.groupby if self.groupby is None else list(self.groupby)

        if not any(col in self.wb_dmos.columns for col in self.INPUT_COLUMNS):
            raise ValueError(
                f"None of the valid input columns {self.INPUT_COLUMNS} found in the passed dataframe."
            )

        if groupby and not all(
            col in self.wb_dmos.reset_index().columns for col in groupby
        ):
            raise ValueError(
                f"Not all groupby columns {self.groupby} found in the passed dataframe."
            )

        data_correct_index = (
            wb_dmos.reset_index()
            .set_index([*(groupby or []), self.unique_wb_id_column])
            .sort_index()
        )

        if not data_correct_index.index.is_unique:
            raise ValueError(
                f"The passed data contains multiple entries for the same groupby columns {groupby}. "
                "Make sure that the passed data in `unique_wb_id_column` is unique for every groupby column "
                "combination."
            )

        if wb_dmos_mask is not None:
            # We silent the warning about downcasting, as we correctly infer the types.
            # This can be removed once we upgrade to pandas 3.0
            with option_context("future.no_silent_downcasting", True):
                wb_dmos_mask = (
                    wb_dmos_mask.fillna(True)
                    .infer_objects(copy=False)
                    .reset_index()
                    .set_index([*(groupby or []), self.unique_wb_id_column])
                    .sort_index()
                )

            if not data_correct_index.index.equals(wb_dmos_mask.index):
                raise ValueError(
                    "The data mask seems to be missing some data indices. "
                    "`wb_dmos_mask` must have exactly the same indices as `wb_dmos` after grouping."
                )

            wb_dmos_mask = wb_dmos_mask.reindex(data_correct_index.index)
            # In case the wb_dmos_mask has columns that are not boolean, we set them to True implicitly
            # Note, in columns with "Falsy" values (e.g. empty string) this might introduce some False values, but
            # this shouldn't matter, as the potential additional columns will not be used in the aggregation anyway.
            wb_dmos_mask = wb_dmos_mask.astype(bool)

            # We remove all individual elements from the data that are flagged as implausible in the data mask.
            self.filtered_wb_dmos_ = data_correct_index.where(wb_dmos_mask)
            # And then we need to consider some special cases:
            if (
                "duration_s" in data_correct_index.columns
                and "duration_s" in wb_dmos_mask.columns
            ):
                # If the duration is implausible, we need to remove the whole walking bout
                self.filtered_wb_dmos_ = self.filtered_wb_dmos_.where(
                    wb_dmos_mask["duration_s"]
                )
            if "walking_speed_mps" in data_correct_index.columns:
                walking_speed_filter = pd.Series(True, index=data_correct_index.index)
                # Walking speed is also implausible, if stride length or cadence are implausible
                if "stride_length_m" in wb_dmos_mask.columns:
                    walking_speed_filter &= wb_dmos_mask["stride_length_m"]
                if "cadence_spm" in wb_dmos_mask.columns:
                    walking_speed_filter &= wb_dmos_mask["cadence_spm"]
                self.filtered_wb_dmos_.loc[:, "walking_speed_mps"] = (
                    self.filtered_wb_dmos_.loc[:, "walking_speed_mps"].where(
                        walking_speed_filter
                    )
                )
        else:
            self.filtered_wb_dmos_ = data_correct_index.copy()

        available_filters_and_aggs = self._select_aggregations(
            data_correct_index.columns
        )
        self.aggregated_data_ = self._apply_aggregations(
            self.filtered_wb_dmos_, groupby, available_filters_and_aggs
        )
        self.aggregated_data_ = self._fillna_count_columns(self.aggregated_data_)
        self.aggregated_data_ = self._convert_units(self.aggregated_data_)

        if self.use_original_names is False:
            self.aggregated_data_ = self.aggregated_data_.rename(
                columns=self.ALTERNATIVE_NAMES, errors="ignore"
            )

        return self

    def _select_aggregations(
        self, data_columns: list[str]
    ) -> list[tuple[str, dict[str, tuple[str, typing.Union[str, typing.Callable]]]]]:
        """
        Select aggregation definitions based on available input columns.

        For each duration filter, the method identifies the subset of aggregation
        mappings that can be computed given the columns present in the input
        DataFrame.

        Returns
        -------
        list
            A list of ``(filter, aggregations)`` tuples defining the aggregations
            that will be applied.
        """
        available_filters_and_aggs = []
        for filt, aggs in self._FILTERS_AND_AGGS:
            if all([filt is not None, "duration_s" not in data_columns]):
                warnings.warn(
                    f"Filter '{filt}' for walking bout length cannot be applied, "
                    "because the data does not contain a 'duration_s' column.",
                    stacklevel=2,
                )
                continue

            # check if the property to aggregate is contained in data columns
            available_aggs = {
                key: value for key, value in aggs.items() if value[0] in data_columns
            }
            if available_aggs:
                available_filters_and_aggs.append((filt, available_aggs))
        return available_filters_and_aggs

    @staticmethod
    def _apply_aggregations(
        filtered_data: pd.DataFrame,
        groupby: typing.Optional[list[str]],
        available_filters_and_aggs: list[
            tuple[str, dict[str, tuple[str, typing.Union[str, typing.Callable]]]]
        ],
    ) -> pd.DataFrame:
        """
        Apply duration filters and aggregation functions.

        For each filter condition, the method subsets the data and performs the
        specified aggregations, either per group (if ``groupby`` is defined) or on
        the entire dataset.
        """
        aggregated_results = []
        for f, agg in available_filters_and_aggs:
            internal_filtered = filtered_data if f is None else filtered_data.query(f)
            if groupby:
                data_to_agg = internal_filtered.groupby(groupby)
            else:
                data_to_agg = internal_filtered.groupby(
                    pd.Series("all_wbs", index=internal_filtered.index)
                )
            aggregated_results.append(data_to_agg.agg(**agg))
        return pd.concat(aggregated_results, axis=1)

    def _fillna_count_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing counts with zero.

        Count-type parameters are NaN when no walking bouts satisfy the corresponding
        duration filter. These values are replaced with zero and cast to nullable
        integer dtype.
        """
        count_columns = [col for col in self._COUNT_COLUMNS if col in data.columns]
        data.loc[:, count_columns] = data.loc[:, count_columns].fillna(0)
        return data.astype(dict.fromkeys(count_columns, "Int64"))

    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply unit conversions to selected aggregated parameters.

        Conversion factors are defined in ``_UNIT_CONVERSIONS`` and applied column-wise.
        """
        for col, factor in self._UNIT_CONVERSIONS.items():
            if col in data.columns:
                data.loc[:, col] *= factor
        return data
