from collections.abc import Iterator, Sequence
from types import MappingProxyType
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
    overload,
)

import pandas as pd
import numpy as np
from tpcp import cf
from tpcp.misc import (
    BaseTypedIterator,
    TypedIteratorResultTuple,
    custom_hash,
    set_defaults,
)
from tpcp.misc._typed_iterator import _NotSet
from typing_extensions import TypeAlias


class Region(NamedTuple):
    """Tuple structure describing a single gait region.

    Represents an identified gait segment using an ID and an interval defined
    by start and end indices. Optionally tracks the origin ID value if provided.
    """

    id: str
    start: int
    end: int
    id_origin: Optional[str] = None


class RegionDataTuple(NamedTuple):
    """Tuple structure representing a gait region together with its data slice.

    Contains the Region definition and a DataFrame containing the associated
    portion of the signal or measurement data.
    """

    region: Region
    data: pd.DataFrame


T = TypeVar("T")
DataclassT = TypeVar("DataclassT")


def _infer_id_col(region_list: pd.DataFrame, id_col: Optional[str] = None) -> str:
    """
    Select the column or index level used as the unique identifier for regions.

    If `id_col` is given, it is returned directly. Otherwise, the function checks
    whether `wb_id` or `gs_id` exists among the columns. If neither is present,
    the function uses a single index level if available. In all other cases,
    an error is raised.

    Parameters
    ----------
    region_list : pd.DataFrame
        Table containing the gait region definitions.
    id_col : str, optional
        Column or index level name used to uniquely identify each returned Region.
        If omitted, the function attempts to determine a suitable name.

    Returns
    -------
    str
        The identifier column or index level name to be used for Region objects.

    Raises
    ------
    ValueError
        If no unambiguous identifier column or index level can be determined.
    """
    if id_col is not None:
        return id_col
    region_list_all_cols = region_list.reset_index().columns
    if "wb_id" in region_list_all_cols:
        return "wb_id"
    if "gs_id" in region_list_all_cols:
        return "gs_id"
    if (
        len(region_list.index.names) == 1
        and (name := region_list.index.names[0]) is not None
    ):
        return name
    raise ValueError(
        "Unable to determine identifier column; please supply the column name explicitly."
    )


def _validate_region_list(
    region_list: pd.DataFrame, id_col: str, data_len: int
) -> None:
    """
    Validate structural and logical consistency of the region definition table.

    Validation steps include:
    - verifying the presence of required columns
    - ensuring numeric (int/float) values without NaNs in start/end
    - ensuring start >= 0 and end >= start
    - enforcing integer conversion of start/end when all values are integer-like
      and verifying that end does not exceed the dataset length in that case

    Parameters
    ----------
    region_list : pd.DataFrame
        Region specification table containing at least `id_col`, `start`, `end`.
    id_col : str
        Identifier column to validate.
    data_len : int
        Total length of the provided dataset used to check end boundaries.

    Raises
    ------
    ValueError, TypeError
        If the table is malformed or contains invalid values.
    """
    # Required columns
    required_cols = [id_col, "start", "end"]
    missing_cols = set(required_cols) - set(region_list.columns)
    if missing_cols:
        raise ValueError(f"Region list missing required columns: {missing_cols}")

    # Numeric check and no NaNs
    if not region_list.empty:
        if not (
            pd.api.types.is_numeric_dtype(region_list["start"])
            and pd.api.types.is_numeric_dtype(region_list["end"])
        ):
            raise TypeError(
                "Columns 'start' and 'end' must be numeric (integers or floats)."
            )
        if region_list[["start", "end"]].isna().any().any():
            raise ValueError("Columns 'start' and 'end' must not contain NaN values.")

        # Logical checks that are independent of integer-vs-float
        if (region_list["start"] < 0).any():
            raise ValueError("Region 'start' indices must be ≥ 0.")
        if (region_list["end"] < region_list["start"]).any():
            raise ValueError("Region 'end' must be greater than or equal to 'start'.")

        # If all values are integer-like we can safely coerce and apply the original end<=data_len check.
        starts = region_list["start"].to_numpy(dtype=float)
        ends = region_list["end"].to_numpy(dtype=float)
        integer_like = np.isclose(starts, np.round(starts), atol=1e-8) & np.isclose(
            ends, np.round(ends), atol=1e-8
        )

        if integer_like.all():
            # safe to coerce to integers in-place (keeps downstream .iloc semantics)
            region_list["start"] = region_list["start"].round().astype(int)
            region_list["end"] = region_list["end"].round().astype(int)
            if (region_list["end"] > data_len).any():
                raise ValueError("Region 'end' exceeds the total length of the data.")


def iter_gs(
    data: pd.DataFrame, region_list: pd.DataFrame, *, id_col: Optional[str] = None
) -> Iterator[tuple[Region, pd.DataFrame]]:
    """
    Iterate through gait regions and yield the corresponding data segments.

    For each region entry, the function produces a tuple containing:
    - the Region object (with ID and start/end markers)
    - the DataFrame slice from `data` corresponding to that interval

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset from which slices are extracted.
    region_list : pd.DataFrame
        Table specifying gait regions using `start` and `end` boundaries.
    id_col : str, optional
        Name of the unique identifier column. If omitted, the identifier is
        inferred automatically.

    Yields
    ------
    tuple[Region, pd.DataFrame]
        Region metadata alongside the extracted data window.

    Notes
    -----
    The returned data preserves the original indices. The first row of the slice
    matches the element located at `data.iloc[start]`.
    """
    # Determine which column to use as unique ID
    index_col = _infer_id_col(region_list, id_col)

    # Reset index to work with columns consistently
    region_list = region_list.reset_index()

    # Validate the region list before iteration
    _validate_region_list(region_list, index_col, len(data))

    relevant_cols = [index_col, "start", "end"]

    # Iterate over each gait-sequence and yield its corresponding data slice
    for gs in region_list[relevant_cols].itertuples(index=False):
        yield RegionDataTuple(Region(*gs, index_col), data.iloc[gs.start : gs.end])


@dataclass
class FullPipelinePerGsResult:
    """
    Default data container for storing per-region pipeline outputs.

    Each instance corresponds to a single gait region and contains several
    result DataFrames, such as initial contacts, cadence, stride length, and
    walking speed.

    Attributes
    ----------
    ic_list : pd.DataFrame
        Initial contact locations relative to the start of each region, stored
        in a column named `ic`.
    cadence_per_sec : pd.DataFrame
        Cadence values within the region.
    stride_length_per_sec : pd.DataFrame
        Per-second stride length values for the region.
    walking_speed_per_sec : pd.DataFrame
        Per-second walking speed values for the region.
    """

    ic_list: pd.DataFrame
    cadence_per_sec: pd.DataFrame
    stride_length_per_sec: pd.DataFrame
    walking_speed_per_sec: pd.DataFrame


def _build_id_cols(region: Region, parent_region: Optional[Region]) -> list[str]:
    iter_index_name = [region.id_origin]
    if parent_region is not None:
        iter_index_name = [parent_region.id_origin, *iter_index_name]
    return iter_index_name


def _validate_iter_type(iter_type: str, parent_region: Optional[Region]) -> None:
    if iter_type not in ["__sub_iter__", "__main__"]:
        raise RuntimeError("Iterator mode not recognised.")
    if parent_region and iter_type == "__main__":
        raise RuntimeError("Main iteration must not have a parent context.")
    if not parent_region and iter_type == "__sub_iter__":
        raise RuntimeError("Sub-iteration requires an enclosing parent region.")


def create_aggregate_df(
    key: str,
    offset_columns: Sequence[str] = ("start", "end"),
    *,
    shift_index: bool = False,
    _null_value: _NotSet = BaseTypedIterator.NULL_VALUE,
) -> Callable[[list["GsIterator.IteratorResult[Any]"]], T][pd.DataFrame]:
    """Create an aggregator for the GS iterator that aggregates dataframe results into a single dataframe.

    The aggregator combines DataFrame outputs for a given result attribute across
    all regions and adjusts specified columns by adding the original region start.
    This produces global reference values relative to the beginning of the full
    recording.

    Parameters
    ----------
    key : str
        Name of the attribute within the result object to aggregate.
    offset_columns : Sequence[str]
        Columns whose values should be shifted to reflect absolute positioning.
        Defaults to ("start", "end").
    shift_index : bool
        If True, index values in each DataFrame are shifted based on the region
        start to align with the full dataset timeline.
    _null_value
        Marker that denotes missing or unavailable results.

    Notes
    -----
    Offsets are computed based on the start of the region. When running nested
    iterations, the parent region context is used to compute the proper offset.
    """

    def aggregate_df(values: list["GsIterator.IteratorResult[Any]"]) -> pd.DataFrame:
        non_null_results: list[GsIterator.IteratorResult[pd.DataFrame]] = (
            GsIterator.filter_iterator_results(values, key, _null_value)
        )
        if len(non_null_results) == 0:
            # Note: We don't have a way to properly know the names of the index cols or the cols themselve here...
            return pd.DataFrame()

        # We assume that all elements have the same iteration context.
        first_element = non_null_results[0]
        iter_index_name = _build_id_cols(
            first_element.input.region,
            first_element.iteration_context.get("parent_region", None),
        )

        to_concat = {}
        for rt in non_null_results:
            df = rt.result
            region_id, offset, *_ = rt.input.region

            parent_region: Optional[Region] = rt.iteration_context.get(
                "parent_region", None
            )

            _validate_iter_type(rt.iteration_name, parent_region)

            if parent_region:
                offset += parent_region.start
                region_id = (parent_region.id, region_id)

            df = df.copy()
            if offset_columns:
                cols_to_fix = set(offset_columns).intersection(df.columns)
                df[list(cols_to_fix)] += offset
            if shift_index:
                df.index += offset
            to_concat[region_id] = df

        return pd.concat(
            to_concat,
            names=[*iter_index_name, *next(iter(to_concat.values())).index.names],
        )

    return aggregate_df


class GsIterator(BaseTypedIterator[RegionDataTuple, DataclassT], Generic[DataclassT]):
    """
    Iterator that segments input data into gait regions and processes them.

    Combines region-based iteration with structured result handling. For each
    region, an instance of a result dataclass is supplied to collect outputs from
    user-defined operations. Aggregation functions allow combining results
    across regions into final outputs.

    Parameters
    ----------
    data_type : type[DataclassT]
        Dataclass specifying the per-region result structure. The default uses
        FullPipelinePerGsResult, suitable for common gait analysis metrics.
    aggregations : Sequence[tuple[str, Callable]]
        Optional list defining aggregation functions per result attribute.
        Specified as a sequence of (attribute name, aggregation function). If no
        aggregator is defined for a key, a list of raw values is returned.

    Class Attributes
    ----------------
    NULL_VALUE
        Default placeholder for unset values in result fields.
    PredefinedParameters
        Configurable presets using FullPipelinePerGsResult for standard pipelines.
    DefaultAggregators
        Factory functions for constructing aggregation functions.
    IteratorResult
        Type alias for iteration result tuples.

    Attributes
    ----------
    results_
        Aggregated output using configured aggregation functions. Available only
        after the iteration completes.
    raw_results_
        List of result entries for each region. Can be used directly for custom
        processing.
    done_
        Dictionary indicating completion state of main and sub-iterations.

    See Also
    --------
    tpcp.misc.BaseTypedIterator
        Base class implementation.
    tpcp.misc.TypedIterator
        Generic typed iterator version.
    iter_gs
        Functional helper for region iteration.
    """

    # This is required to correctly interfere the new bound type
    IteratorResult: TypeAlias = TypedIteratorResultTuple[RegionDataTuple, DataclassT]

    class PredefinedParameters:
        """Predefined parameters for the gait-sequence iterator.

        Attributes
        ----------
        default_aggregation
            The default of the TypedIterator using the :class:`FullPipelinePerGsResult` as data_type and trying to
            aggregate all results so that the time values in the final outputs are relative to the start of the
            recording.
        default_aggregation_rel_to_gs
            Same as ``default_aggregation``, but the time values in the final outputs are relative to the start of the
            respective gait-sequence (i.e. no modification of the time values is done).

        """

        default_aggregation: Final = MappingProxyType(
            {
                "data_type": FullPipelinePerGsResult,
                "aggregations": cf(
                    [
                        ("ic_list", create_aggregate_df("ic_list", ["ic"])),
                        (
                            "cadence_per_sec",
                            create_aggregate_df(
                                "cadence_per_sec", [], shift_index=True
                            ),
                        ),
                        (
                            "stride_length_per_sec",
                            create_aggregate_df(
                                "stride_length_per_sec", [], shift_index=True
                            ),
                        ),
                        (
                            "walking_speed_per_sec",
                            create_aggregate_df(
                                "walking_speed_per_sec", [], shift_index=True
                            ),
                        ),
                    ]
                ),
            }
        )
        default_aggregation_rel_to_gs: Final = MappingProxyType(
            {
                "data_type": FullPipelinePerGsResult,
                "aggregations": cf(
                    [
                        ("ic_list", create_aggregate_df("ic_list", [])),
                        ("cadence_per_sec", create_aggregate_df("cadence_per_sec", [])),
                        (
                            "stride_length_per_sec",
                            create_aggregate_df("stride_length", []),
                        ),
                        (
                            "walking_speed_per_sec",
                            create_aggregate_df("gait_speed", []),
                        ),
                    ]
                ),
            }
        )

    class DefaultAggregators:
        """Available aggregators for the gait-sequence iterator.

        Note, that all of them are constructors for aggregators, as they have some configuration options.
        To use them as aggregators, you need to call them with the desired configuration.
        """

        create_aggregate_df = create_aggregate_df

    # We provide this explicit overload, so that the type of the default value is correcttly inferred.
    # This way there is not need to "bind" FullPipelinePerGsResult on init, when the defaults are used.
    @overload
    def __init__(
        self: "GsIterator[FullPipelinePerGsResult]",
        data_type: type[FullPipelinePerGsResult] = ...,
        aggregations: Sequence[tuple[str, Callable[[list[IteratorResult]], Any]]] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: "GsIterator[DataclassT]",
        data_type: type[DataclassT] = ...,
        aggregations: Sequence[tuple[str, Callable[[list[IteratorResult]], Any]]] = ...,
    ) -> None: ...

    @set_defaults(**PredefinedParameters.default_aggregation)
    def __init__(
        self,
        data_type,
        aggregations,
    ) -> None:
        super().__init__(data_type, aggregations)

    def iterate(
        self, data: pd.DataFrame, region_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[Region, pd.DataFrame], DataclassT]]:
        """
        Iterate through all gait regions sequentially.

        Parameters
        ----------
        data : pd.DataFrame
            Input data table.
        region_list : pd.DataFrame
            Region definitions using `start` and `end`, expressed in the same index
            units as `data`.

        Yields
        ------
        region_data : tuple[Region, pd.DataFrame]
            Region definition and the corresponding data window.
        result_object
            Fresh dataclass instance used to record outputs for the region.
        """
        yield from self._iterate(iter_gs(data, region_list))

    def iterate_subregions(
        self, sub_region_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[Region, pd.DataFrame], DataclassT]]:
        """
        Iterate over subregions within the active parent gait region.

        Produces region/data pairs where the start and end markers are defined
        relative to the current parent region.

        Parameters
        ----------
        sub_region_list : pd.DataFrame
            Subregion definitions relative to the current region.

        Returns
        -------
        region_data : tuple[Region, pd.DataFrame]
            Subregion definition with the corresponding data slice.
        result_object
            New result container for the subregion outputs.
        """
        # We only allow sub iterations, when there are no other subiterations running.
        if getattr(self, "done_", {}).get("__main__", True):
            raise ValueError(
                "Cannot start sub-iterations after main iteration has completed."
            )
        if not self.done_.get("__sub_iter__", True):
            raise ValueError("Sub-iterations are not allowed within sub-iterations.")

        current_result = self._raw_results[-1]
        current_region, current_data = current_result.input

        # We calculate the hash of the last outer result to check if it was changed during the sub-iteration.
        # Note, that when you are using the ``subregion`` context manager, this check is duplicated.
        # The reason for that is that with the context manager, we have a clear entry and exist point that we would
        # not otherwise have, when we simply iterate a single subregion.
        current_result_obj = current_result.result
        before_result_hash = custom_hash(current_result_obj)

        yield from self._iterate(
            iter_gs(current_data, sub_region_list),
            iteration_name="__sub_iter__",
            iteration_context={"parent_region": current_region},
        )

        after_result_hash = custom_hash(current_result_obj)
        if before_result_hash != after_result_hash:
            raise RuntimeError(
                "Detected modifications to the outer-result during sub-iteration; avoid mutating outer state. "
                "This might lead to unexpected results. "
                "Make sure you use the result object returned by the subregion iteration."
            )

    def with_subregion(
        self, sub_region_list: pd.DataFrame
    ) -> tuple[tuple[Region, pd.DataFrame], DataclassT]:
        """Retrieve a single subregion from the active parent region.

        Accepts a region list containing exactly one subregion, executes iteration
        for that subregion, and returns the result immediately. For multiple
        subregions, use `iterate_subregions`.

        Parameters
        ----------
        sub_region_list : pd.DataFrame
            Single-row region list describing one subregion relative to the current
            region boundaries.

        Returns
        -------
        inputs : tuple[Region, pd.DataFrame]
            Subregion definition and related data slice.
        result_object
            Dataclass instance to collect results for the subregion.

        Notes
        -----
        Implemented internally using `iterate_subregions`, but completes iteration of
        the single subregion before returning.
        """
        if len(sub_region_list) != 1:
            raise ValueError(
                "``with_subregions`` can only be used with single-subregions. "
                "However, the passed ``region_list`` has 0 or more than one GSs. "
                "If you want to process multiple sub-regions, use ``iterate_subregions``."
            )
        return list(self.iterate_subregions(sub_region_list))[0]  # noqa: RUF015

    @contextmanager
    def subregion(
        self, sub_region_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[Region, pd.DataFrame], DataclassT]]:
        """Context manager to process a single subregion of the active region.

        Provides a controlled block to operate on a subregion and ensures that only
        the returned result object is modified. Detects unwanted mutations of the
        parent result during the block.

        Parameters
        ----------
        sub_region_list : pd.DataFrame
            Region list with one subregion, relative to the current region.

        Yields
        ------
        inputs : tuple[Region, pd.DataFrame]
            Region metadata and data slice.
        result_object
            Result container for the subregion.

        Notes
        -----
        Effectively a convenience wrapper over `with_subregion`, adding integrity
        checks against modifying the parent result in error.
        """
        outer_result = self._raw_results[-1].result
        before_result_hash = custom_hash(outer_result)

        try:
            yield self.with_subregion(sub_region_list)
        finally:
            after_result_hash = custom_hash(outer_result)
            if before_result_hash != after_result_hash:
                raise RuntimeError(
                    "It looks like you accessed the old result object of the main iteration within the subregion "
                    "context. "
                    "Use the result object returned by the context manager!"
                )
