"""We use the Mobilise-D logic with amended and simplified rules"""

import warnings
from collections.abc import Hashable
from itertools import count
from types import MappingProxyType
from typing import Final, Optional

import numpy as np
import pandas as pd
from tpcp import Algorithm, cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from multigait.pipeline.wba_base import BaseWbRule, EndOfStrideList

from multigait.pipeline.utils._wb_criteria import BreakCriteria, StridesCriteria


class WbAssembly(Algorithm):
    """
    Assemble strides into walking bouts (WBs) based on configurable rules.

    This algorithm groups individual strides into walking bouts using a two-step approach:
    1. Iterates through strides to create preliminary walking bouts, stopping when a termination
       rule is triggered.
    2. Applies inclusion rules to determine which preliminary bouts are valid for final output.

    Parameters
    ----------
    rules : Optional[list[tuple[str, BaseWbRule]]]
        Rules to guide WB assembly. Each rule is provided as a tuple containing a name and an
        instance of a subclass of `BaseWbRule`.
        If `None`, all strides are returned as a single walking bout.

    Attributes
    ----------
    annotated_stride_list_ : pd.DataFrame
        All strides assigned to a WB, indexed by `wb_id` and stride ID. Can be used to compute
        per-WB statistics.
    wbs_ : dict
        Dictionary of final walking bouts. Keys are WB IDs; values are DataFrames of associated strides.
    wb_meta_parameters_ : pd.DataFrame
        Summary of WB-level metadata (start, end, duration, stride count, etc.).
    excluded_stride_list_ : pd.DataFrame
        Strides not included in any WB, either discarded during preliminary WB formation or excluded
        by inclusion rules.
    excluded_wbs_ : dict
        Walking bouts that were discarded by inclusion rules.
    termination_reasons_ : pd.DataFrame
        Reason each WB was terminated. Columns: `rule_name`, `rule_obj`.
    exclusion_reasons_ : pd.DataFrame
        Reason each WB was excluded. Columns: `rule_name`, `rule_obj`.
    stride_exclusion_reasons_ : pd.DataFrame
        Reason individual strides were excluded from WBs.

    Other Parameters
    ----------------
    filtered_stride_list : pd.DataFrame
        Strides provided to the `assemble` method.
    raw_initial_contacts : Optional[pd.DataFrame]
        Raw initial contact timestamps from the IC detection algorithm, used to compute
        `n_raw_initial_contacts` per WB.
    sampling_rate_hz : float
        Sampling frequency of the data, used to convert stride start/end indices to seconds.

    Notes
    -----
    - Each rule can act as both a termination and inclusion rule using `check_wb_start_end` and
      `check_include`.
    - Termination rules can adjust the start and end of a WB, while inclusion rules simply
      determine if a WB should be kept.
    - The algorithm chooses the rule that results in the shortest WB at each step.
    """

    _action_methods = ("assemble",)
    _composite_params = ("rules",)

    rules: Optional[list[tuple[str, BaseWbRule]]]

    filtered_stride_list: pd.DataFrame
    sampling_rate_hz: float

    annotated_stride_list_: pd.DataFrame
    excluded_stride_list_: pd.DataFrame
    termination_reasons_: pd.DataFrame
    exclusion_reasons_: pd.DataFrame
    stride_exclusion_reasons_: pd.DataFrame
    wb_meta_parameters_: pd.DataFrame
    wbs_: dict[Hashable, pd.DataFrame]
    excluded_wbs_: dict[Hashable, pd.DataFrame]

    _wb_id_map: dict[str, int]

    class PredefinedParameters:
        mobilised: Final = MappingProxyType(
            {
                "rules": [
                    (
                        "min_strides",
                        StridesCriteria(min_strides=4),
                    ),
                    ("max_break", BreakCriteria(max_break_s=3)),
                ]
            }
        )

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.mobilised.items()})
    def __init__(self, rules: Optional[list[tuple[str, BaseWbRule]]]) -> None:
        self.rules = rules

    @property
    def wbs_(self) -> dict[Hashable, pd.DataFrame]:
        if len(self.annotated_stride_list_) == 0:
            return {}
        return {
            k: v.reset_index("wb_id", drop=True)
            for k, v in self.annotated_stride_list_.groupby("wb_id")
        }

    @property
    def wb_meta_parameters_(self) -> pd.DataFrame:
        if len(self.annotated_stride_list_) == 0:
            columns = ["start", "end", "n_strides", "duration_s"]
            if self.raw_initial_contacts is not None:
                columns.append("n_raw_initial_contacts")
            return pd.DataFrame(columns=columns, index=pd.Index([], name="wb_id"))

        n_strides = (
            self.annotated_stride_list_.groupby("wb_id").size().rename("n_strides")
        )
        parameters = self.termination_reasons_.loc[n_strides.index]
        start_end = self.annotated_stride_list_.groupby("wb_id").agg(
            {"start": "min", "end": "max"}
        )

        df = (
            pd.concat([start_end, n_strides, parameters], axis=1)
            .assign(
                duration_s=lambda x: (x["end"] - x["start"]) / self.sampling_rate_hz
            )
            .astype({"start": int, "end": int, "n_strides": int, "duration_s": float})
        )
        if self.raw_initial_contacts is not None:
            n_initial_contacts = pd.Series(index=df.index, dtype="Int64")
            for wb_id, wb in df.iterrows():
                n_initial_contacts[wb_id] = self.raw_initial_contacts[
                    self.raw_initial_contacts["ic"].between(wb["start"], wb["end"])
                ].shape[0]
            df["n_raw_initial_contacts"] = n_initial_contacts

        return df

    @property
    def excluded_wbs_(self) -> dict[str, pd.DataFrame]:
        if len(self.excluded_stride_list_) == 0:
            return {}
        return {
            k: v.reset_index("pre_wb_id", drop=True)
            # We group only the strides that are part of a preliminary WB
            for k, v in self.excluded_stride_list_[
                self.excluded_stride_list_.index.get_level_values("pre_wb_id").notna()
            ].groupby("pre_wb_id")
        }

    def assemble(
        self,
        filtered_stride_list: pd.DataFrame,
        *,
        raw_initial_contacts: Optional[pd.DataFrame] = None,
        sampling_rate_hz: float,
    ) -> Self:
        """
        Assemble final walking bouts from a pre-filtered list of strides.

        Parameters
        ----------
        filtered_stride_list : pd.DataFrame
            Strides to assemble into WBs. Must include `start` and `end` columns (in samples).
            Additional columns may be required depending on the rules.
        raw_initial_contacts : Optional[pd.DataFrame]
            Optional initial contact timestamps, used to count raw ICs per WB.
        sampling_rate_hz : float
            Sampling frequency of the data in Hz. Used to convert stride indices to seconds
            if necessary.

        Returns
        -------
        Self
            Returns the `WbAssembly` instance with populated WB attributes:
            `annotated_stride_list_`, `excluded_stride_list_`, `wbs_`, `excluded_wbs_`,
            `termination_reasons_`, `exclusion_reasons_`, and `stride_exclusion_reasons_`.
        """
        for _, rule in self.rules or []:
            if not isinstance(rule, BaseWbRule):
                raise TypeError(
                    "All rules must be instances of `WBCriteria` or one of its child classes."
                )

        self.filtered_stride_list = filtered_stride_list
        self.raw_initial_contacts = raw_initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        stride_list_sorted = self.filtered_stride_list.sort_values(by=["start", "end"])

        (
            preliminary_wb_list,
            excluded_wb_list,
            exclusion_reasons,
            termination_reasons,
            excluded_strides,
            stride_exclusion_reasons,
        ) = self._apply_termination_rules(stride_list_sorted)
        wb_list, excluded_wb_list_2, exclusion_reasons_2 = self._apply_inclusion_rules(
            preliminary_wb_list
        )
        # After we have the final wbs, we rewrite the wb_ids to be easier to read.
        self._wb_id_map = {k: i for i, k in enumerate(wb_list.keys())}
        stride_index_col_name = filtered_stride_list.index.names
        new_index_cols = ["wb_id", *stride_index_col_name]
        if len(wb_list) > 0:
            self.annotated_stride_list_ = pd.concat(
                wb_list, names=new_index_cols
            ).reset_index()
            self.annotated_stride_list_["wb_id"] = self.annotated_stride_list_[
                "wb_id"
            ].map(self._wb_id_map)
        else:
            self.annotated_stride_list_ = pd.DataFrame(
                columns=[*filtered_stride_list.columns, *new_index_cols]
            )
        self.annotated_stride_list_ = self.annotated_stride_list_.set_index(
            new_index_cols
        )

        pre_id_index_cols = ["pre_wb_id", *stride_index_col_name]
        if (
            len(
                combined_excluded_stride_list := {
                    **excluded_wb_list,
                    **excluded_wb_list_2,
                }
            )
            > 0
        ):
            excluded_strides_in_wbs = pd.concat(
                combined_excluded_stride_list, names=pre_id_index_cols
            ).reset_index()
        else:
            excluded_strides_in_wbs = pd.DataFrame(
                columns=[*filtered_stride_list.columns, *pre_id_index_cols]
            )
        other_excluded_strides = excluded_strides.assign(pre_wb_id=None).reset_index()
        with warnings.catch_warnings():
            # We ignore Pandas Future Warning here, as we actually want the new behaviour, but there is no way to
            # enable it.
            warnings.simplefilter("ignore", category=FutureWarning)
            self.excluded_stride_list_ = pd.concat(
                [excluded_strides_in_wbs, other_excluded_strides]
            ).set_index(pre_id_index_cols)
        self.termination_reasons_ = pd.DataFrame.from_dict(
            termination_reasons, orient="index", columns=["rule_name", "rule_obj"]
        ).rename_axis(index="wb_id")
        # For the WBs that "made" it into the final structure, we need to rename the wb_ids
        self.termination_reasons_.index = self.termination_reasons_.index.map(
            self._wb_id_map
        )
        self.exclusion_reasons_ = pd.DataFrame.from_dict(
            {**exclusion_reasons, **exclusion_reasons_2},
            orient="index",
            columns=["rule_name", "rule_obj"],
        ).rename_axis(index="wb_id")
        self.stride_exclusion_reasons_ = pd.DataFrame.from_dict(
            stride_exclusion_reasons, orient="index", columns=["rule_name", "rule_obj"]
        ).rename_axis(index=filtered_stride_list.index.name)
        return self

    def _apply_termination_rules(
        self, stride_list: pd.DataFrame
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, pd.DataFrame],
        dict[str, tuple[str, BaseWbRule]],
        dict[str, tuple[str, BaseWbRule]],
        pd.DataFrame,
        dict[str, tuple[str, BaseWbRule]],
    ]:
        id_counter = count(start=1)
        end = 0
        preliminary_wb_list = {}
        termination_reasons = {}
        excluded_wb_list = {}
        exclusion_reasons = {}
        excluded_strides = []
        stride_exclusion_reasons = {}
        last_end = -1
        while end < len(stride_list):
            start = end
            (
                final_start,
                final_end,
                restart_at,
                start_delay_reason,
                termination_reason,
            ) = self._find_first_preliminary_wb(stride_list, start)
            preliminary_wb_id = f"pre_{next(id_counter)}"

            if final_start > final_end:
                # There was a termination criteria, but the WB was never properly started
                final_start = final_end + 1
                excluded_wb_list[preliminary_wb_id] = stride_list.iloc[
                    start:final_start
                ]
                exclusion_reasons[preliminary_wb_id] = start_delay_reason
                termination_reasons[preliminary_wb_id] = termination_reason
            else:
                # The preliminary WB is saved
                preliminary_wb_list[preliminary_wb_id] = stride_list.iloc[
                    final_start : final_end + 1
                ]
                termination_reasons[preliminary_wb_id] = termination_reason
                # Save strides that were excluded in the beginning as an excluded strides
                removed_strides = stride_list.iloc[start:final_start]
                if len(removed_strides) > 0:
                    for s in removed_strides.index:
                        stride_exclusion_reasons[s] = start_delay_reason
                    excluded_strides.append(removed_strides)
                # Save strides that were excluded in the end as an excluded strides
                removed_strides = stride_list.iloc[final_end + 1 : restart_at]
                if len(removed_strides) > 0:
                    for s in removed_strides.index:
                        stride_exclusion_reasons[s] = termination_reason
                    excluded_strides.append(removed_strides)
                # Save strides that were not considered a valid start of a WB.
                # I.e. all strides since the end of the last WB until the original start of the WB
                if start > last_end + 1:
                    removed_strides = stride_list.iloc[last_end + 1 : start]
                    for s in removed_strides.index:
                        stride_exclusion_reasons[s] = ("no_start", None)
                    excluded_strides.append(removed_strides)

                last_end = restart_at - 1

            end = restart_at
        if len(excluded_strides) > 0:
            excluded_strides = pd.concat(excluded_strides)
        else:
            excluded_strides = pd.DataFrame(columns=stride_list.columns)
        return (
            preliminary_wb_list,
            excluded_wb_list,
            exclusion_reasons,
            termination_reasons,
            excluded_strides,
            stride_exclusion_reasons,
        )

    def _find_first_preliminary_wb(
        self,
        stride_list: pd.DataFrame,
        original_start: int,
    ) -> tuple[
        int,
        int,
        int,
        Optional[tuple[str, BaseWbRule]],
        Optional[tuple[str, BaseWbRule]],
    ]:
        end_index = len(stride_list)
        current_end = original_start
        current_start = original_start
        start_delay_reason = None
        # We loop until the actual end value (so one more than the actual existing stride indices).
        # The reason is, that we want to give rules the chance to check the end of a WB including the final stride of
        # the list.
        while current_end <= end_index:
            (
                tmp_start,
                tmp_end,
                tmp_restart_at,
                tmp_start_delay_reason,
                termination_reason,
            ) = self._check_wb_start_end(
                stride_list, original_start, current_start, current_end
            )
            if termination_reason:
                # In case of termination return directly.
                # Do not consider further updates on the start as they might be made based on strides that are not part
                # of a WB
                return (
                    current_start,
                    tmp_end,
                    tmp_restart_at,
                    start_delay_reason,
                    termination_reason,
                )
            if tmp_start_delay_reason:
                start_delay_reason = tmp_start_delay_reason
                current_start = tmp_start

            current_end += 1
        # In case we end the loop without any termination rule firing
        return (
            current_start,
            len(stride_list),
            len(stride_list),
            start_delay_reason,
            ("end_of_list", EndOfStrideList()),
        )

    def _check_wb_start_end(
        self,
        stride_list: pd.DataFrame,
        original_start: int,
        current_start: int,
        current_end: int,
    ) -> tuple[
        int,
        int,
        int,
        Optional[tuple[str, BaseWbRule]],
        Optional[tuple[str, BaseWbRule]],
    ]:
        termination_rule = None
        start_delay_rule = None
        tmp_start = -1
        tmp_end = np.inf
        tmp_restart_at = np.inf
        for rule_name, rule in self.rules or []:
            start, end, restart_at = rule.check_wb_start_end(
                stride_list,
                original_start=original_start,
                current_start=current_start,
                current_end=current_end,
                sampling_rate_hz=self.sampling_rate_hz,
            )
            if start is not None and start > tmp_start:
                tmp_start = start
                start_delay_rule = (rule_name, rule)
            if end is not None and end < tmp_end:
                tmp_end = end
                tmp_restart_at = restart_at
                termination_rule = (rule_name, rule)
        return tmp_start, tmp_end, tmp_restart_at, start_delay_rule, termination_rule

    def _apply_inclusion_rules(
        self, preliminary_wb_list: dict[str, pd.DataFrame]
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, pd.DataFrame],
        dict[str, tuple[str, BaseWbRule]],
    ]:
        wb_list = {}
        removed_wb_list = {}
        exclusion_reasons = {}
        for wb_id, stride_list in preliminary_wb_list.items():
            for rule_name, rule in self.rules or []:
                if not rule.check_include(
                    stride_list, sampling_rate_hz=self.sampling_rate_hz
                ):
                    removed_wb_list[wb_id] = stride_list
                    exclusion_reasons[wb_id] = (rule_name, rule)
                    break
            else:
                wb_list[wb_id] = stride_list
        return wb_list, removed_wb_list, exclusion_reasons
