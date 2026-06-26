from __future__ import annotations

import typing
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from pandas import option_context
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack
from src.multigait.aggregation._aggregator_base import AggregatorBase


def _custom_quantile(x: pd.Series) -> float:
    """Calculate the 90th percentile of the passed data."""
    if x.isna().all():
        return np.nan
    return np.nanpercentile(x, 90)


def _coefficient_of_variation(x: pd.Series) -> float:
    """Calculate variation of the passed data."""
    return x.std() / x.mean()


class LaboratoryAggregator(AggregatorBase):
    """Aggregation algorithm for laboratory analyses in Multimobility.

    This aggregator computes aggregated digital mobility outcomes (DMOs) across
    all walking bouts, without any duration-based subgrouping. Its behaviour
    follows that of `GenericAggregator` regarding input validation, masking,
    unit conversion, and alternative naming.

    Aggregated metrics computed:

    - Count of walking bouts: ``*_all_sum`` (e.g., ``wb_all_sum``)
    - Total walking duration in minutes: ``walkdur_all_sum`` (converted from seconds)
    - Mean / median / sum of walking-bout variables across all bouts:
        - Duration: ``wbdur_all_avg`` (median), ``wbdur_all_var`` (across-bout variance),
          ``wbdur_all_p90`` (90th percentile)
        - Stride duration: ``strdur_all_avg`` (mean), ``strdur_all_var`` (across-bout variance)
        - Cadence: ``cadence_all_avg`` (mean), ``cadence_all_var`` (across-bout variance)
        - Walking speed: ``ws_all_avg`` (mean), ``ws_all_p90`` (90th percentile)
        - Stride length: ``strlen_all_avg`` (mean), ``strlen_all_var`` (across-bout variance)
    - Mean of per-bout variability metrics (within-bout):
        - Coefficient of variation (CV): ``*_wb_cv`` (e.g., ``strdur_all_wb_cv``)
        - RMSSD: ``*_wb_rmssd`` (mean of per-bout RMSSD, e.g., ``strlen_all_wb_rmssd``)

    Parameters
    ----------
    groupby : list[str] | None
        Columns to group the data by. For laboratory analyses, this is typically
        per participant (``["participant_id"]``) or None for a single recording.
    unique_wb_id_column : str
        Name of the column uniquely identifying each walking bout within a group.
    use_original_names : bool
        If True, the aggregator maps internal descriptive names to legacy names
        as defined in ``ALTERNATIVE_NAMES``.

    Other Parameters
    ----------------
    wb_dmos_mask : pd.DataFrame, optional
        Boolean mask of same shape as ``wb_dmos`` indicating plausibility of each
        measure. NaNs are interpreted as True (plausible). Special semantics:
        - ``duration_s``: if False, the entire bout is removed
        - ``walking_speed_mps``: ignored if False
        - ``stride_length_m``: ignored with walking speed if False
        - ``cadence_spm``: ignored with walking speed if False
        - ``stride_duration_s``: ignored if False

    Attributes
    ----------
    aggregated_data_ : pd.DataFrame
        Aggregated DMO results for each group.
    filtered_wb_dmos_ : pd.DataFrame
        Input DMOs after removal of implausible measurements according to
        ``wb_dmos_mask``. Index includes ``groupby`` columns and ``unique_wb_id_column``.
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
        "ws_all_avg": "wb_all__walking_speed_mps__avg",
        "strlen_all_avg": "wb_all__stride_length_m__avg",
        "ws_all_p90": "wb_all__walking_speed_mps__p90",
        "strlen_all_var": "wb_all__stride_length_m__var",
        "ws_all_wb_cv": "wb_all__walking_speed_mps__cv__avg",
        "strlen_all_wb_cv": "wb_all__stride_length_m__cv__avg",
        "ws_all_wb_rmssd": "wb_all__walking_speed_mps__rmssd__avg",
        "strlen_all_wb_rmssd": "wb_all__stride_length_m__rmssd__avg",
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

    # Aggregations computed once for all walking bouts (added ws/strlen entries)
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
        "ws_all_avg": ("walking_speed_mps", "mean"),
        "strlen_all_avg": ("stride_length_m", "mean"),
        "ws_all_p90": ("walking_speed_mps", _custom_quantile),
        "strlen_all_var": ("stride_length_m", _coefficient_of_variation),
        "ws_all_wb_cv": ("walking_speed_mps_cv", "mean"),
        "strlen_all_wb_cv": ("stride_length_m_cv", "mean"),
        "ws_all_wb_rmssd": ("walking_speed_mps_rmssd", "mean"),
        "strlen_all_wb_rmssd": ("stride_length_m_rmssd", "mean"),
        "alpha": ("alpha", "mean"),
    }

    # Only the 'all WBs' aggregation is used
    _FILTERS_AND_AGGS: typing.ClassVar[
        list[
            tuple[
                typing.Optional[str],
                dict[str, tuple[str, typing.Union[str, typing.Callable]]],
            ]
        ]
    ] = [
        (None, _ALL_WB_AGGS),
    ]

    _UNIT_CONVERSIONS: typing.ClassVar = {
        "walkdur_all_sum": 1 / 60,
    }

    # Only keep count columns relevant for the "all" aggregation
    _COUNT_COLUMNS: typing.ClassVar = [
        "wb_all_sum",
        "wbsteps_all_sum",
    ]

    groupby: typing.Optional[typing.Sequence[str]]
    unique_wb_id_column: str

    wb_dmos_mask: pd.DataFrame

    filtered_wb_dmos_: pd.DataFrame

    class PredefinedParameters:
        multimobility_data: Final = MappingProxyType(
            {
                # In laboratory analyses we aggregate per participant only
                "groupby": ["participant_id"],
                "unique_wb_id_column": "wb_id",
                "use_original_names": True,
            }
        )

        single_recording: Final = MappingProxyType(
            {
                "groupby": None,
                "unique_wb_id_column": "wb_id",
                "use_original_names": False,
            }
        )

    @set_defaults(
        **{k: cf(v) for k, v in PredefinedParameters.single_recording.items()}
    )
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
        """Aggregate per the BaseAggregator contract (see docstring of the class)."""
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
            wb_dmos_mask = wb_dmos_mask.astype(bool)

            # Remove individual elements flagged as implausible
            self.filtered_wb_dmos_ = data_correct_index.where(wb_dmos_mask)

            # If duration is implausible, remove whole walking bout
            if (
                "duration_s" in data_correct_index.columns
                and "duration_s" in wb_dmos_mask.columns
            ):
                self.filtered_wb_dmos_ = self.filtered_wb_dmos_.where(
                    wb_dmos_mask["duration_s"]
                )

            # If stride_length or cadence implausible, walking_speed is implausible
            if "walking_speed_mps" in data_correct_index.columns:
                walking_speed_filter = pd.Series(True, index=data_correct_index.index)
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

        # Select aggregations that can be computed from the available columns
        available_filters_and_aggs = self._select_aggregations(
            data_correct_index.columns
        )

        # Apply aggregations (only the single "all WBs" aggregation will be applied)
        self.aggregated_data_ = self._apply_aggregations(
            self.filtered_wb_dmos_, groupby, available_filters_and_aggs
        )

        # Post-process counts and units
        self.aggregated_data_ = self._fillna_count_columns(self.aggregated_data_)
        self.aggregated_data_ = self._convert_units(self.aggregated_data_)

        if self.use_original_names is False:
            self.aggregated_data_ = self.aggregated_data_.rename(
                columns=self.ALTERNATIVE_NAMES, errors="ignore"
            )

        return self

    def _select_aggregations(
        self, data_columns: list[str]
    ) -> list[
        tuple[
            typing.Optional[str],
            dict[str, tuple[str, typing.Union[str, typing.Callable]]],
        ]
    ]:
        """Build list of available aggregations based on columns in the input data.

        Since this class computes only the 'all WBs' aggregations, this simply filters
        _ALL_WB_AGGS by available source columns.
        """
        available_aggs = {
            key: value
            for key, value in self._ALL_WB_AGGS.items()
            if value[0] in data_columns
        }
        if not available_aggs:
            return []
        return [(None, available_aggs)]

    @staticmethod
    def _apply_aggregations(
        filtered_data: pd.DataFrame,
        groupby: typing.Optional[list[str]],
        available_filters_and_aggs: list[
            tuple[str, dict[str, tuple[str, typing.Union[str, typing.Callable]]]]
        ],
    ) -> pd.DataFrame:
        """Apply aggregations to the data. This mirrors GenericAggregator behaviour but only
        the 'all WBs' aggregation is expected to be applied."""
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
        if not aggregated_results:
            # Return empty DataFrame with no columns if nothing to aggregate
            return pd.DataFrame()
        return pd.concat(aggregated_results, axis=1)

    def _fillna_count_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replace NaN count values with 0 and set Int64 dtype for those columns."""
        count_columns = [col for col in self._COUNT_COLUMNS if col in data.columns]
        if count_columns:
            data.loc[:, count_columns] = data.loc[:, count_columns].fillna(0)
            data = data.astype(dict.fromkeys(count_columns, "Int64"))
        return data

    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert the units of the aggregated data to the desired output units."""
        for col, factor in self._UNIT_CONVERSIONS.items():
            if col in data.columns:
                data.loc[:, col] *= factor
        return data
