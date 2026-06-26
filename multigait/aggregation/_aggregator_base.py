from typing import Any
from typing_extensions import Self, Unpack

import pandas as pd
from tpcp import Algorithm


class AggregatorBase(Algorithm):
    """
    Base class for all aggregation implementations.

    Subclasses must implement the `aggregate` method. This method receives
    walking-bout–level digital mobility outputs (DMOs) and produces a
    consolidated, group-level result. The final aggregated output must be
    stored in the attribute `aggregated_data_`.

    The recommended pattern is to define configuration parameters directly
    on the estimator rather than passing them dynamically to `aggregate`.
    Keyword arguments are accepted to support controlled extensions, but
    subclasses should explicitly validate their use.

    Parameters passed to `aggregate`
    --------------------------------
    wb_dmos : pandas.DataFrame
        A table of walking-bout DMOs, with one row per bout and one column
        per DMO variable. Identifiers or metadata (for example participant
        ID, visit date, or bout ID) may be represented as columns or via
        the index.

    Attributes
    ----------
    aggregated_data_ : pandas.DataFrame
        A DataFrame containing the aggregated results. Its index should
        represent the grouping scheme (for example participant and visit).
    """

    _action_methods = ("aggregate",)

    # Input reference for typing
    wb_dmos: pd.DataFrame

    # Output reference for typing
    aggregated_data_: pd.DataFrame

    def aggregate(
        self,
        wb_dmos: pd.DataFrame,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Run the aggregation.

        This method must be implemented in subclasses. It is expected to
        compute the aggregate metrics, assign the result to
        `self.aggregated_data_`, and return the estimator instance.

        Parameters
        ----------
        wb_dmos : pandas.DataFrame
            Walking-bout DMO data as described in the class documentation.

        Returns
        -------
        self : AggregatorBase
            The instance containing the computed aggregation results.
        """
        raise NotImplementedError("Subclasses must implement `aggregate`.")


__all__ = ["AggregatorBase"]
