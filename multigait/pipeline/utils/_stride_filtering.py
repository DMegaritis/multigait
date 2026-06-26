import logging
import warnings
from types import MappingProxyType
from typing import Final, Literal, Optional

import pandas as pd
from tpcp import Algorithm, cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from multigait.pipeline.wba_base import BaseIntervalCriteria, IntervalDurationCriteria

logger = logging.getLogger(__name__)


class StrideFiltering(Algorithm):
    """Selects strides based on a set of criteria.

    - computes and exposes a boolean pass_mask_ (True == stride passes all rules),

    Public attributes:
      - filtered_stride_list_ : DataFrame of strides that passed all rules
      - excluded_stride_list_ : DataFrame of strides that failed at least one rule
      - exclusion_reasons_     : DataFrame with first failing rule for excluded strides
      - check_results_         : DataFrame of booleans per rule
      - pass_mask_             : Boolean Series (True if stride passed all applied rules)

    The defaults (PredefinedParameters.mobilised) remain:
      - duration between 0.2 and 3.0 s

    Note: stride_length_m values are automatically clipped to the [0.15, 2.0] m range
    before applying the rules: values below 0.15 are set to 0.15, values above 2.0
    are set to 2.0. This is implemented as the wrist-based SL estimators are intensity based and have some outliers.
    These outliers are either high is acceleration is high or low if acceleration is low. Hence, according to their rationale,
    we consider that if acceleration is large then stride length should also be large and vice versa. Some outlier are observed in [1] hence correcting them makes sense.

    Behaviour notes:
    - stride_length_m outliers are clipped for downstream computations
    - pass_mask_ and exclusion_reasons_ are determined only by the stride duration

    [1] Megaritis D, Alcock L, Scott K, Hiden H, Cereatti A, Vogiatzis I, Del Din S.
    Real-World Wrist-Derived Digital Mobility Outcomes in People with Multiple Long-Term Conditions: A Comparison of Algorithms.
    Bioengineering (Basel). 2025 Oct 15;12(10):1108. doi: 10.3390/bioengineering12101108. PMID: 41155107; PMCID: PMC12561645.
    """

    _action_methods = ("filter",)
    _composite_params = ("rules",)

    rules: Optional[list[tuple[str, BaseIntervalCriteria]]]
    incompatible_rules: Literal["raise", "warn", "skip"]

    stride_list: pd.DataFrame
    sampling_rate_hz: float

    _exclusion_reasons: pd.DataFrame
    check_results_: pd.DataFrame
    pass_mask_: pd.Series

    class PredefinedParameters:
        mobilised: Final = MappingProxyType(
            {
                "rules": [
                    (
                        "stride_duration_thres",
                        IntervalDurationCriteria(
                            min_duration_s=0.2, max_duration_s=3.0
                        ),
                    ),
                ],
                "incompatible_rules": "warn",
            }
        )

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.mobilised.items()})
    def __init__(
        self,
        rules: Optional[list[tuple[str, BaseIntervalCriteria]]],
        *,
        incompatible_rules: Literal["raise", "warn", "skip"],
    ) -> None:
        self.incompatible_rules = incompatible_rules
        self.rules = rules

    @property
    def filtered_stride_list_(self) -> pd.DataFrame:
        # Return only strides that passed all rules (copy to avoid accidental modifications)
        return self.stride_list[self.pass_mask_].copy()

    @property
    def excluded_stride_list_(self) -> pd.DataFrame:
        return self.stride_list[~self.pass_mask_].copy()

    @property
    def exclusion_reasons_(self) -> pd.DataFrame:
        # Only return rows where a rule excluded the stride
        return self._exclusion_reasons[self._exclusion_reasons["rule_name"].notna()]

    def filter(self, stride_list: pd.DataFrame, *, sampling_rate_hz: float) -> Self:
        """Filter the stride list.

        Parameters
        ----------
        stride_list
            The stride list to filter. Each row represents a stride. Must contain at least 'start' and 'end'.
        sampling_rate_hz
            Sampling rate (Hz). If start/end are already in seconds, set to 1.

        Returns
        -------
        self
            Instance with check_results_, pass_mask_, exclusion_reasons_ and related attributes set.
        """
        # Validate rules are of expected type
        for _, rule in self.rules or []:
            if not isinstance(rule, BaseIntervalCriteria):
                raise TypeError(
                    "All rules must be instances of `IntervalSummaryCriteria` or one of its child classes."
                )

        # Keep an internal copy to avoid accidental external mutation
        self.stride_list = stride_list.copy()
        self.sampling_rate_hz = sampling_rate_hz

        # Correct extreme stride_length_m values:
        # - values < 0.15 -> 0.15
        # - values > 2.0  -> 2.0
        # Clip stride_length_m for downstream use.
        if "stride_length_m" in self.stride_list.columns:
            original = self.stride_list["stride_length_m"]
            clipped = original.clip(lower=0.15, upper=2.0)
            changed_mask = ~original.eq(clipped)
            if changed_mask.any():
                n_changed = int(changed_mask.sum())
                logger.warning(
                    "StrideFiltering: clipped %d stride_length_m values to the [0.15, 2.0] m range.",
                    n_changed,
                )
                # Apply clipped values only where they changed
                self.stride_list.loc[changed_mask, "stride_length_m"] = clipped[
                    changed_mask
                ]

        # If there are no rules or empty input, create trivial outputs (all pass)
        if self.rules is None or len(self.rules) == 0 or stride_list.empty:
            # All strides pass
            self.pass_mask_ = pd.Series(True, index=stride_list.index)
            self._exclusion_reasons = pd.DataFrame(
                columns=["rule_name", "rule_obj"]
            ).reindex(stride_list.index)
            self.check_results_ = pd.DataFrame(index=stride_list.index)
            return self

        rules_as_dict = dict(self.rules)
        stride_list_cols = set(self.stride_list.columns)

        # Run each compatible rule and collect boolean Series
        rule_results: dict[str, pd.Series] = {}
        for name, rule in rules_as_dict.items():
            compatible = set(rule.requires_columns()).issubset(stride_list_cols)
            if not compatible:
                msg = f"Rule {name} requires columns {rule.requires_columns()} which are not present in the stride list."
                if self.incompatible_rules == "raise":
                    raise ValueError(msg)
                if self.incompatible_rules == "warn":
                    warnings.warn(msg + " Skipping rule.", stacklevel=1)
                    logger.warning(msg + " Skipping rule.")
                # if "skip", do nothing
                continue
            rule_results[name] = rule.check_multiple(
                self.stride_list, sampling_rate_hz=sampling_rate_hz
            )

        # If no rules could be applied, mark all as pass
        if len(rule_results) == 0:
            self.pass_mask_ = pd.Series(True, index=stride_list.index)
            self._exclusion_reasons = pd.DataFrame(
                columns=["rule_name", "rule_obj"]
            ).reindex(stride_list.index)
            self.check_results_ = pd.DataFrame(index=stride_list.index)
            return self

        # Boolean DataFrame: one column per rule
        self.check_results_ = pd.concat(rule_results, axis=1)

        # Mask of strides that pass all applied rules
        self.pass_mask_ = self.check_results_.all(axis=1)

        def _get_rule_obj(rule_names: pd.Series) -> pd.Series:
            with pd.option_context("future.no_silent_downcasting", True):
                return rule_names.replace(rules_as_dict).infer_objects(copy=False)

        # Determine first failing rule for each stride. idxmin returns first False position;
        # replace rows that passed all rules with NaN.
        self._exclusion_reasons = (
            self.check_results_.idxmin(axis=1)
            .where(~self.pass_mask_)  # keep NaN when all True
            .rename("rule_name")
            .to_frame()
            .assign(rule_obj=lambda df_: df_["rule_name"].pipe(_get_rule_obj))
        )

        return self
