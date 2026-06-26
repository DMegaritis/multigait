from typing import Any
import copy

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack


class BaseGsdDetector(Algorithm):
    """
    Minimal standalone BaseGsdDetector.

    This class is intentionally small: it keeps the tpcp.Algorithm contract and
    provides a convenient clone() helper. Implementations of concrete GSD
    algorithms should subclass this and implement `detect`.
    """

    _action_methods = ("detect",)

    # expected attributes for type checkers / docs
    data: pd.DataFrame
    sampling_rate_hz: float
    gs_list_: pd.DataFrame

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Implement in subclass."""
        raise NotImplementedError

    def clone(self) -> "BaseGsdDetector":
        """Return a deep copy of this detector so callers can do clone().detect(...)."""
        return copy.deepcopy(self)

    @staticmethod
    def empty_gs_df() -> pd.DataFrame:
        """Return an empty gait-sequence dataframe with the expected layout."""
        df = pd.DataFrame(columns=["start", "end"])
        df.index.name = "gs_id"
        return df


__all__ = ["BaseGsdDetector"]
