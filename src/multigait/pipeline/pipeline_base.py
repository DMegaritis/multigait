"""base typing interface for pipelines."""

from typing import Generic, Optional, TypeVar
import pandas as pd
from tpcp import Pipeline
from src.multigait.data.base_data import BaseGaitDataset

GaitDatasetT = TypeVar("GaitDatasetT", bound=BaseGaitDataset)


class PipelineBase(Pipeline[GaitDatasetT], Generic[GaitDatasetT]):
    """
    Minimal typed base class for pipelines for pipeline.
    """

    # final per-stride dataframe (interpolated stride-level parameters for selected strides)
    stride_params_df: pd.DataFrame

    # per-walking-bout aggregated parameters
    wb_params_df: pd.DataFrame

    # mask or validity dataframe for per-wb parameters (optional; may be None if not used)
    wb_param_mask: Optional[pd.DataFrame]

    # aggregated/device-level or recording-level results (optional)
    aggregated_df: Optional[pd.DataFrame]

    # Any concrete pipeline class should implement this to declare the dataset type it expects.
    # Keep as an abstract placeholder to remind implementors to bind the concrete dataset type.
    def expected_dataset_type(self) -> type:
        """Return the dataset class this pipeline expects (subclass of BaseGaitDataset)."""
        raise NotImplementedError
