from typing import Any
import copy

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack


class BaseIcDetector(Algorithm):
    """
    Minimal standalone BaseIcDetector.

    Subclasses must implement:
        detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs) -> Self

    The detect implementation should:
      - set self.data
      - set self.sampling_rate_hz
      - set self.ic_list_ (pandas.DataFrame with column "ic" and index name "step_id")
      - return self

    This base class adds a clone() helper that returns a deep copy of the detector instance.
    """

    _action_methods = ("detect",)

    # expected attributes for type checkers / docs
    data: pd.DataFrame
    sampling_rate_hz: float
    ic_list_: pd.DataFrame

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Implement in subclass."""
        raise NotImplementedError

    def clone(self) -> "BaseIcDetector":
        """Return a deep copy of this detector so callers can do clone().detect(...)."""
        return copy.deepcopy(self)


__all__ = ["BaseIcDetector"]
