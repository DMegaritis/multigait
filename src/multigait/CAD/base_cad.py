from typing import Any
import copy

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack


class BaseCadDetector(Algorithm):
    """Minimal standalone BaseCadDetector.

    This base class mirrors the style of other small base classes in the codebase:
      - Declares the action method name so tpcp-based code can call it generically.
      - Documents expected attributes for implementers.
      - Provides a clone() helper that returns a deep copy of the instance.
    """

    _action_methods = ("calculate",)

    # expected attributes for type checkers / docs
    data: pd.DataFrame
    sampling_rate_hz: float
    initial_contacts: pd.DataFrame
    cadence_per_sec_: pd.DataFrame

    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: pd.DataFrame,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Implement in subclass."""
        raise NotImplementedError

    def clone(self) -> "BaseCadDetector":
        """Return a deep copy of this detector so callers can do clone().calculate(...)."""
        return copy.deepcopy(self)


__all__ = ["BaseCadDetector"]
