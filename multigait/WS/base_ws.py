"""Minimal BaseWsDetector that inherits from tpcp.Algorithm.

Intended to be a lightweight drop-in base class for walking-speed calculators that:
- provides clone(); can do: detector.clone().calculate(...)
- keeps the tpcp.Algorithm integration (action methods tuple) so it can be used where Algorithm-derived classes are expected

"""

from typing import Any, Optional
import copy

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack


class BaseWsDetector(Algorithm):
    """Minimal standalone BaseWsDetector.

    This base class mirrors the style of other minimal detector bases in this codebase:
      - Declares the action method name so tpcp-based code can call it generically.
      - Documents expected attributes for implementers.
      - Provides a clone() helper that returns a deep copy of the instance.
    """

    _action_methods = ("calculate",)

    # expected attributes for type checkers / docs
    data: pd.DataFrame
    initial_contacts: Optional[pd.DataFrame]
    sampling_rate_hz: float
    cadence_per_sec: Optional[pd.DataFrame]
    stride_length_per_sec: Optional[pd.DataFrame]

    # results
    walking_speed_per_sec_: pd.DataFrame

    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: Optional[pd.DataFrame] = None,
        cadence_per_sec: Optional[pd.DataFrame] = None,
        stride_length_per_sec: Optional[pd.DataFrame] = None,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Implement in subclass: compute walking_speed_per_sec_ and return self."""
        raise NotImplementedError

    def clone(self) -> "BaseWsDetector":
        """Return a deep copy of this detector so callers can do clone().calculate(...)."""
        return copy.deepcopy(self)


__all__ = ["BaseWsDetector"]
