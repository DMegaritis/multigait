from typing import Optional, Tuple
import pandas as pd
import numpy as np
from tpcp import BaseTpcpObject


def _check_thresh(
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    allow_both_none: bool = False,
) -> Tuple[float, float]:
    if not allow_both_none and lower_threshold is None and upper_threshold is None:
        raise ValueError("You need to provide at least an upper or a lower threshold.")
    if lower_threshold is None:
        lower_threshold = -np.inf
    if upper_threshold is None:
        upper_threshold = np.inf
    if not lower_threshold < upper_threshold:
        raise ValueError(
            f"The lower threshold must be below the upper threshold. Currently: {lower_threshold} not < {upper_threshold}"
        )
    return lower_threshold, upper_threshold


def _compare_with_thresh(
    value: float,
    lower_threshold: float,
    upper_threshold: float,
    inclusive: tuple[bool, bool],
) -> bool:
    lower_threshold, upper_threshold = _check_thresh(lower_threshold, upper_threshold)
    lower_op = np.greater_equal if inclusive[0] else np.greater
    upper_op = np.less_equal if inclusive[1] else np.less
    return bool(lower_op(value, lower_threshold) and upper_op(value, upper_threshold))


def _compare_with_thresh_multiple(
    values: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
    inclusive: tuple[bool, bool],
) -> pd.Series:
    lower_threshold, upper_threshold = _check_thresh(lower_threshold, upper_threshold)
    lower_op = np.greater_equal if inclusive[0] else np.greater
    upper_op = np.less_equal if inclusive[1] else np.less
    return pd.Series(
        (lower_op(values, lower_threshold) & upper_op(values, upper_threshold)).astype(
            bool
        ),
        index=values.index,
    )


class BaseWbRule:
    """
    Minimal base for WB criteria. Implements the interface expected by WbAssembly.

    Child classes should override `check_wb_start_end` for termination logic
    or `check_include` for inclusion logic.
    """

    def check_wb_start_end(
        self,
        stride_list: pd.DataFrame,
        *,
        original_start: int,
        current_start: int,
        current_end: int,
        sampling_rate_hz: Optional[float] = None,
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Decide whether the current stride should be included in the WB.
        By default, allows all strides without adjustment.
        """
        return None, None, current_end

    def check_include(
        self,
        stride_list: pd.DataFrame,
        *,
        sampling_rate_hz: Optional[float] = None,
    ) -> bool:
        """
        Decide whether the preliminary WB should be included as a final WB.
        By default, always includes it.
        """
        return True


class EndOfStrideList(BaseWbRule):
    """Dummy criteria used internally to represent the end of the stride list."""

    pass


class BaseIntervalCriteria(BaseTpcpObject):
    """Base class for criteria that filter intervals (e.g., strides)."""

    def check_multiple(
        self, intervals: pd.DataFrame, *, sampling_rate_hz: Optional[float] = None
    ) -> pd.Series:
        """Return a boolean Series: True if intervals meet the criterion."""
        raise NotImplementedError("Must be implemented by child class.")

    def requires_columns(self) -> list[str]:
        """Return the list of required columns in the interval DataFrame."""
        raise NotImplementedError("Must be implemented by child class.")


class IntervalDurationCriteria(BaseIntervalCriteria):
    """Checks the interval duration computed as (end - start) / sampling_rate_hz."""

    _START_COL_NAME = "start"
    _END_COL_NAME = "end"

    def __init__(
        self,
        min_duration_s: Optional[float] = None,
        max_duration_s: Optional[float] = None,
        inclusive: tuple[bool, bool] = (False, True),
    ):
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.inclusive = inclusive

    def requires_columns(self) -> list[str]:
        return [self._START_COL_NAME, self._END_COL_NAME]

    def _get_values(
        self, intervals: pd.DataFrame, sampling_rate_hz: Optional[float] = None
    ) -> pd.Series:
        if sampling_rate_hz is None:
            raise ValueError(
                "sampling_rate_hz must be provided for IntervalDurationCriteria"
            )
        try:
            return (
                intervals[self._END_COL_NAME] - intervals[self._START_COL_NAME]
            ) / sampling_rate_hz
        except KeyError as e:
            raise ValueError(
                f"Intervals must contain both '{self._START_COL_NAME}' and '{self._END_COL_NAME}'"
            ) from e

    def check_multiple(
        self, intervals: pd.DataFrame, *, sampling_rate_hz: Optional[float] = None
    ) -> pd.Series:
        values = self._get_values(intervals, sampling_rate_hz)
        return _compare_with_thresh_multiple(
            values, self.min_duration_s, self.max_duration_s, self.inclusive
        )
