import warnings
from types import MappingProxyType
from typing import Any, Final, Generic, Optional
import pandas as pd
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self


# Multimobility imports
from multigait.CAD.cad import Cadence
from multigait.CAD.base_cad import BaseCadDetector
from multigait.GSD.GSD3 import KheirkhahanGSD
from multigait.GSD.base_gsd import BaseGsdDetector
from multigait.ICD.ICD2 import McCamleyIC
from multigait.ICD.base_ic import BaseIcDetector
from multigait.SL.SL1 import WeinbergSL
from multigait.SL.base_sl import BaseSlDetector
from multigait.WS.walking_speed import Ws
from multigait.WS.base_ws import BaseWsDetector

# Multimobility function imports
from multigait.pipeline.utils.ic_to_stride import strides_list_from_ic_list_no_lrc
from multigait.pipeline.utils._wb_assembly import WbAssembly
from multigait.pipeline.utils._thresholds import get_thresholds, apply_thresholds
from multigait.aggregation._generic_aggregator import GenericAggregator
from multigait.pipeline.utils._stride_filtering import StrideFiltering
from multigait.pipeline.iterator import GsIterator, FullPipelinePerGsResult
from multigait.pipeline.utils._operations import create_multi_groupby
from multigait.utils.interp import map_seconds_to_regions
from multigait.aggregation._aggregator_base import AggregatorBase
from multigait.pipeline.pipeline_base import PipelineBase
from multigait.pipeline.pipeline_base import GaitDatasetT
from multigait.utils.data_conversions import rename_axes_to_body
from multigait.pipeline.utils._var_dmos import within_wb_var
from multigait.pipeline.utils.alpha import compute_alpha_mle


# Expected variability DMO columns (keep in sync with within_wb_var output)
VAR_DMO_COLUMNS = [
    "stride_duration_s_cv",
    "stride_duration_s_rmssd",
    "cadence_spm_cv",
    "cadence_spm_rmssd",
    "stride_length_m_cv",
    "stride_length_m_rmssd",
    "walking_speed_mps_cv",
    "walking_speed_mps_rmssd",
]

class MultimobilityPipeline(PipelineBase[GaitDatasetT], Generic[GaitDatasetT]):
    """
    Multimobility pipeline for wrist-worn devices. This pipeline is based on the MobGap pipeline
    structure and is designed to process raw IMU data to compute stride- and walking-bout-level
    gait parameters and daily mobility outcomes (DMOs).

    The pipeline can be instantiated either with custom algorithm instances or with predefined
    parameters for gait sequence detection, stride metrics calculation, and aggregation.

    Pipeline workflow
    -----------------
    1. Detect gait sequences in a recording.
    2. Detect initial contacts within each gait sequence.
    3. Compute per-second gait parameters: cadence, stride length, and walking speed.
    4. Convert per-second outputs to per-stride parameters using detected initial contacts.
    5. Filter strides and assemble walking bouts (WBs).
    6. Compute per-WB aggregated parameters and within-WB variability DMOs.
    7. Optionally apply physiological thresholds and aggregate WB-level DMOs.

    Notes
    -----
    - Concrete algorithm instances (objects implementing the detector/calculator interfaces)
      are passed to the constructor; no specific implementations are required at class definition.
    - All major pipeline results are stored as attributes after `run()` execution.

    Parameters
    ----------
    gait_sequence_detection : BaseGSD
        Algorithm instance for gait sequence detection.
    initial_contact_detection : BaseIC
        Algorithm instance for initial contact detection.
    cadence_calculation : Optional[BaseCadence], default=None
        Algorithm instance for cadence calculation (per-second).
    stride_length_calculation : Optional[BaseSL], default=None
        Algorithm instance for stride length calculation (per-second).
    walking_speed_calculation : Optional[BaseWS], default=None
        Algorithm instance for walking speed calculation (per-second).
    stride_selection : BaseStrideSelection
        Algorithm instance used to filter/select strides.
    wba : BaseWbAssembly
        Logic to assemble filtered strides into walking bouts.
    dmo_thresholds : Optional[dict], default=None
        Thresholds for DMO computation, e.g., physiological thresholds.
    dmo_aggregation : Optional[BaseAggregator], default=None
        Aggregator instance to compute aggregated DMOs from per-WB results.

    Raises
    ------
    ValueError
        If the input datapoint lacks required metadata (participant_metadata).

    Attributes
    ----------
    per_stride_parameters_ : pd.DataFrame
        Final per-stride parameters after stride selection and WBA.
    per_wb_parameters_ : pd.DataFrame
        Aggregated parameters per walking bout.
    aggregated_parameters_ : pd.DataFrame
        Aggregated daily mobility outcomes (DMOs) across the dataset.
    gait_sequence_detection_ : BaseGSD
        Instance used for gait sequence detection.
    initial_contact_detection_ : BaseIC
        Instance used for initial contact detection.
    cadence_calculation_ : BaseCadence
        Instance used for cadence calculation.
    stride_length_calculation_ : BaseSL
        Instance used for stride length calculation.
    walking_speed_calculation_ : BaseWS
        Instance used for walking speed calculation.
    stride_selection_ : BaseStrideSelection
        Instance used for stride filtering/selection.
    wba_ : BaseWbAssembly
        Instance used for whole-bout assembly.
    dmo_thresholds_ : dict
        Thresholds used internally for DMO computation.
    dmo_aggregation_ : BaseAggregator
        Aggregator used to compute DMOs.
    gs_list_ : list
        Detected gait sequences (start/end times).
    gs_iterator_ : iterator
        Iterator over gait sequences for intermediate pipeline processing.
    per_wb_parameter_mask_ : pd.Series
        Boolean mask indicating valid strides per walking bout.
    raw_ic_list_ : list
        Raw initial contact events detected from IMU data.
    raw_per_stride_parameters_ : pd.DataFrame
        Raw stride-wise parameters before filtering and stride selection.
    raw_per_sec_parameters_ : pd.DataFrame
        Raw per-second gait parameters.
    var_dmos : pd.DataFrame
        Within-WB variability DMOs computed from per-stride parameters.

    Examples
    --------
    >>> pipeline = MultimobilityPipeline(
    >>>     gait_sequence_detection=GSDAlgorithm(),
    >>>     initial_contact_detection=ICAlgorithm(),
    >>>     cadence_calculation=CadenceAlgorithm(),
    >>>     stride_length_calculation=StrideLengthAlgorithm(),
    >>>     walking_speed_calculation=WalkingSpeedAlgorithm(),
    >>>     stride_selection=StrideSelection(),
    >>>     wba=WalkingBoutAssembly(),
    >>>     dmo_thresholds=my_thresholds,
    >>>     dmo_aggregation=DMOAggregator()
    >>> )
    >>> pipeline.run(datapoint=my_dataset)
    >>> print(pipeline.aggregated_parameters_)
    """

    gait_sequence_detection: BaseGsdDetector
    initial_contact_detection: BaseIcDetector
    cadence_calculation: Optional[BaseCadDetector]
    stride_length_calculation: Optional[BaseSlDetector]
    walking_speed_calculation: Optional[BaseWsDetector]
    stride_selection: StrideFiltering
    wba: WbAssembly
    dmo_thresholds: Optional[pd.DataFrame]
    dmo_aggregation: AggregatorBase

    datapoint: GaitDatasetT

    # Algos with results
    gait_sequence_detection_: BaseGsdDetector
    gs_iterator_: GsIterator[FullPipelinePerGsResult]
    stride_selection_: StrideFiltering
    wba_: WbAssembly
    dmo_aggregation_: Optional[AggregatorBase]

    # Intermediate results
    gs_list_: pd.DataFrame
    raw_ic_list_: pd.DataFrame
    raw_per_sec_parameters_: pd.DataFrame
    raw_per_stride_parameters_: pd.DataFrame

    _all_action_kwargs: dict[str, Any]

    class PredefinedParameters:
        preliminary_multimobility: Final = MappingProxyType(
            {
                "gait_sequence_detection": KheirkhahanGSD(),
                "initial_contact_detection": McCamleyIC(),
                "cadence_calculation": Cadence(),
                "stride_length_calculation": WeinbergSL(),
                "walking_speed_calculation": Ws(),
                "stride_selection": StrideFiltering(),
                "wba": WbAssembly(),
                "dmo_thresholds": get_thresholds(),
                "dmo_aggregation": GenericAggregator(**GenericAggregator.PredefinedParameters.single_day),
            }
        )


    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsdDetector,
        initial_contact_detection: BaseIcDetector,
        cadence_calculation: Optional[BaseCadDetector],
        stride_length_calculation: Optional[BaseSlDetector],
        walking_speed_calculation: Optional[BaseWsDetector],
        stride_selection: StrideFiltering,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: Optional[AggregatorBase],
    ) -> None:
        self.gait_sequence_detection = gait_sequence_detection
        self.initial_contact_detection = initial_contact_detection
        self.cadence_calculation = cadence_calculation
        self.stride_length_calculation = stride_length_calculation
        self.walking_speed_calculation = walking_speed_calculation
        self.stride_selection = stride_selection
        self.wba = wba
        self.dmo_thresholds = dmo_thresholds
        self.dmo_aggregation = dmo_aggregation


    def run(self, datapoint: GaitDatasetT, **kwargs) -> Self:
        """
        Run the full pipeline on a single datapoint.

        Parameters
        ----------
        datapoint : GaitDatasetT
            A single-row dataset object representing one recording. Required attributes:
              - participant_metadata (dict-like): participant information used by some algorithms (e.g. foot length).
              - recording_metadata (dict-like): recording-level metadata.
              - sampling_rate_hz (float): sampling frequency in Hz.
              - data_ss (pd.DataFrame): sensor signals (will be axis-renamed to match expectations).
              - group_label: optional grouping label.

        Returns
        -------
        self
            The same pipeline instance with results attached to its attributes as described in the class docstring.

        Raises
        ------
        ValueError
            If the datapoint does not expose participant_metadata (required for default stride-length and thresholds).
        """
        try:
            participant_metadata = datapoint.participant_metadata
        except AttributeError as e:
            raise ValueError(
                "The provided dataset does not provide any participant metadata. "
                "For the default algorithms, metadata is required for the ``stride_length_calculation`` "
                "and ``dmo_thresholds`` step. "
                "If you want to use this pipeline without metadata, please provide custom algorithms and"
                "at least implement the ``participant_metadata`` attribute on your dataset, even if it"
                "just returns an empty dictionary."
            ) from e


        self.datapoint = datapoint

        self._all_action_kwargs = {
            **participant_metadata,
            **datapoint.recording_metadata,
            "dp_group": datapoint.group_label,
            "sampling_rate_hz": datapoint.sampling_rate_hz,
        }

        imu_data = rename_axes_to_body(datapoint.data_ss)
        sampling_rate_hz = datapoint.sampling_rate_hz

        self.gait_sequence_detection_ = self.gait_sequence_detection.clone().detect(imu_data)
        self.gs_list_ = self.gait_sequence_detection_.gs_list_
        self.gs_iterator_ = self._run_per_gs(self.gs_list_, imu_data)

        results = self.gs_iterator_.results_

        self.raw_per_sec_parameters_ = pd.concat(
            [
                results.cadence_per_sec,
                results.stride_length_per_sec,
                results.walking_speed_per_sec,
            ],
            axis=1,
        )

        if self.raw_per_sec_parameters_.empty:
            expected_results = [
                calc
                for calc, available in [
                    ("cadence_per_sec", self.cadence_calculation),
                    ("stride_length_per_sec", self.stride_length_calculation),
                    ("walking_speed_per_sec", self.walking_speed_calculation),
                ]
                if available
            ]
            index_names = ["gs_id", "sec_center_samples"]
            self.raw_per_sec_parameters_ = pd.DataFrame(columns=[*expected_results, *index_names]).set_index(
                index_names
            )

        if "r_gs_id" in self.raw_per_sec_parameters_.index.names:
            self.raw_per_sec_parameters_ = self.raw_per_sec_parameters_.reset_index(
                "r_gs_id",
                drop=True,
            )

        if (ic_list := results.ic_list).empty:
            index_names = ["gs_id", "step_id"]
            empty_index = pd.MultiIndex.from_tuples([], names=index_names)
            ic_list = pd.DataFrame(columns=["ic"], index=empty_index)
        self.raw_ic_list_ = ic_list
        self.raw_per_stride_parameters_ = self._sec_to_stride(
            self.raw_per_sec_parameters_, self.raw_ic_list_, sampling_rate_hz
        )

        flat_index = pd.Index(
            ["_".join(str(e) for e in s_id) for s_id in self.raw_per_stride_parameters_.index], name="s_id"
        )
        raw_per_stride_parameters = self.raw_per_stride_parameters_.reset_index("gs_id").rename(
            columns={"gs_id": "original_gs_id"}
        )
        raw_per_stride_parameters.index = flat_index

        self.stride_selection_ = self.stride_selection.clone().filter(
            raw_per_stride_parameters, sampling_rate_hz=sampling_rate_hz
        )
        self.wba_ = self.wba.clone().assemble(
            self.stride_selection_.filtered_stride_list_,
            raw_initial_contacts=ic_list,
            sampling_rate_hz=sampling_rate_hz,
        )

        self.per_stride_parameters_ = self.wba_.annotated_stride_list_
        self.per_wb_parameters_ = self._aggregate_per_wb(self.per_stride_parameters_, self.wba_.wb_meta_parameters_)

        # Variability DMOs calculation
        self.var_dmos = within_wb_var(self.per_stride_parameters_)

        # Ensure the variability DMO DataFrame contains the expected columns even if empty,
        # and aligns with per_wb_parameters_ index so we can safely concat later.
        if self.var_dmos.empty:
            # create an empty dataframe with expected columns and same index as per_wb_parameters_
            self.var_dmos = pd.DataFrame(index=self.per_wb_parameters_.index, columns=VAR_DMO_COLUMNS)

        # Variability DMOs append to per_wb_parameters_
        self.per_wb_parameters_ = pd.concat([self.per_wb_parameters_, self.var_dmos], axis=1)

        # drop temporary or object columns if present
        if "rule_obj" in self.per_wb_parameters_.columns:
            self.per_wb_parameters_ = self.per_wb_parameters_.drop(columns="rule_obj")

        # Alpha calculation
        # Alpha is only relevant in the aggregated results. We compute it on a copy of the per-wb table
        # and we do NOT store it in self.per_wb_parameters_ so the per-wb output does not contain alpha.
        per_wb_with_alpha = compute_alpha_mle(self.per_wb_parameters_.copy())

        if self.dmo_thresholds is None:
            self.per_wb_parameter_mask_ = None
        else:
            height_m = datapoint.participant_metadata.get("height_m")
            if height_m is None:
                height_m = 1.65
                print(f" Warning: height_m not provided for {datapoint.participant_id}, using default value of 1.65m")

            self.per_wb_parameter_mask_ = apply_thresholds(
                self.per_wb_parameters_,
                self.dmo_thresholds,
                height_m=height_m,
            )

        if self.dmo_aggregation is None:
            self.aggregated_parameters_ = None
            return self

        # Use the per-wb table that includes alpha for aggregation, but we keep self.per_wb_parameters_ alpha-free.
        self.dmo_aggregation_ = self.dmo_aggregation.clone().aggregate(
            per_wb_with_alpha, wb_dmos_mask=self.per_wb_parameter_mask_
        )
        self.aggregated_parameters_ = self.dmo_aggregation_.aggregated_data_

        del self._all_action_kwargs
        return self

    def _run_per_gs(
        self,
        gait_sequences: pd.DataFrame,
        imu_data: pd.DataFrame,
    ) -> GsIterator:
        """
        Execute per-gait-sequence processing and return an iterator with results.

        For each gait sequence this method:
          - clones and runs the initial_contact_detection on the GS data,
          - runs cadence, stride length and walking speed calculators if provided,
          - populates fields on the per-GS result object (r) that are later concatenated.

        Parameters
        ----------
        gait_sequences : pd.DataFrame
            DataFrame with detected gait sequences (expected columns: start, end, gs_id or similar).
        imu_data : pd.DataFrame
            Full recording sensor data (time x channels); the function iterates over the GS slices.

        Returns
        -------
        GsIterator
            The iterator object containing per-gs results in its .results_ attribute after iteration.
        """

        gs_iterator = GsIterator[FullPipelinePerGsResult]()

        for (_, gs_data), r in gs_iterator.iterate(imu_data, gait_sequences):
            icd = self.initial_contact_detection.clone().detect(gs_data)
            r.ic_list = icd.ic_list_

            cad_r = None
            if self.cadence_calculation:
                cad = self.cadence_calculation.clone().calculate(
                    gs_data,
                    initial_contacts=icd.ic_list_,
                )
                cad_r = cad.cadence_per_sec_
                r.cadence_per_sec = cad_r

            sl_r = None
            if self.stride_length_calculation:
                sl = self.stride_length_calculation.clone().calculate(
                    gs_data, initial_contacts=icd.ic_list_,
                    **self._all_action_kwargs
                )
                sl_r = sl.stride_length_per_sec_
                r.stride_length_per_sec = sl.stride_length_per_sec_

            if self.walking_speed_calculation:
                ws = self.walking_speed_calculation.clone().calculate(
                    gs_data,
                    initial_contacts=icd.ic_list_,
                    cadence_per_sec=cad_r,
                    stride_length_per_sec=sl_r
                )
                r.walking_speed_per_sec = ws.walking_speed_per_sec_

        return gs_iterator

    def _sec_to_stride(
            self, sec_level_paras: pd.DataFrame, ic_list: pd.DataFrame, sampling_rate_hz: float
    ) -> pd.DataFrame:
        """
        Convert per-second parameters to per-stride parameter regions.

        Behaviour:
          - If ic_list is empty, a stride list with correct index/columns is created via helper.
          - If strides exist, compute stride duration and map per-second parameters into stride regions
            using naive_sec_paras_to_regions.
          - If no strides are present, returns an empty DataFrame with appropriate columns.

        Parameters
        ----------
        sec_level_paras : pd.DataFrame
            Per-second parameters indexed by (gs_id, sec_center_samples) or equivalent.
        ic_list : pd.DataFrame
            Concatenated initial contact detections with at least gs_id, step_id, start, end fields.
        sampling_rate_hz : float
            Recording sampling frequency used to compute durations and convert indices.

        Returns
        -------
        pd.DataFrame
            Per-stride DataFrame with stride-level parameters and a MultiIndex identifying strides.
        """
        if ic_list.empty:
            # We still call the function to get the correct index
            # We need to do that in a separate step, as the groupby is not working with an empty dataframe
            stride_list = strides_list_from_ic_list_no_lrc(ic_list)
        else:
            stride_list = ic_list.groupby("gs_id", group_keys=False).apply(strides_list_from_ic_list_no_lrc)

        stride_list = stride_list.assign(stride_duration_s=lambda df_: (df_.end - df_.start) / sampling_rate_hz)

        # If there are no strides, return empty dataframe with correct columns
        if stride_list.empty:
            # Ensure the returned empty dataframe contains columns from sec_level_paras as your original except-block did.
            stride_list = stride_list.reindex(columns=[*stride_list.columns, *sec_level_paras.columns])
            return stride_list

        # If there are strides we join per-second params into stride regions
        stride_list = create_multi_groupby(
            stride_list,
            sec_level_paras,
            "gs_id",
            group_keys=False,
        ).apply(map_seconds_to_regions, sampling_rate_hz=sampling_rate_hz)

        return stride_list

    def _aggregate_per_wb(self, per_stride_parameters: pd.DataFrame, wb_meta_parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-stride parameters into per-walking-bout (WB) features.

        The function currently computes the mean of a predefined set of stride-level parameters
        per WB and concatenates these aggregated values with WB metadata.

        Parameters
        ----------
        per_stride_parameters : pd.DataFrame
            Per-stride annotated parameters (must include columns in params_to_aggregate and a 'wb_id').
        wb_meta_parameters : pd.DataFrame
            Metadata about walking bouts (indexed by wb_id).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by wb_id containing WB metadata plus aggregated stride-parameter means.
        """
        params_to_aggregate = [
            "stride_duration_s",
            "cadence_spm",
            "stride_length_m",
            "walking_speed_mps",
        ]
        return pd.concat(
            [
                wb_meta_parameters,
                per_stride_parameters.reindex(columns=params_to_aggregate)
                .groupby(["wb_id"])
                .mean(),
            ],
            axis=1,
        )

class MultimobilityPipelineSuggested(MultimobilityPipeline[GaitDatasetT], Generic[GaitDatasetT]):
    """This pipeline is constructed with a predefined set of algorithms which exhibited the best performance in [1].

    This subclass wraps MultimobilityPipeline and provides a set of predefined algorithm instances
    (see PredefinedParameters.preliminary_multimobility) that reflect the configuration
    reported to perform well in our evaluations.

    The constructor accepts the same parameters as MultimobilityPipeline, but default values
    are applied via the set_defaults decorator so users can instantiate it without specifying
    every argument.

    .. [1] Megaritis D, Alcock L, Scott K, Hiden H, Cereatti A, Vogiatzis I, Del Din S.
    Real-World Wrist-Derived Digital Mobility Outcomes in People with Multiple Long-Term Conditions: A Comparison of Algorithms.
    Bioengineering (Basel). 2025 Oct 15;12(10):1108. doi: 10.3390/bioengineering12101108. PMID: 41155107; PMCID: PMC12561645.
    """

    @set_defaults(**{k: cf(v) for k, v in MultimobilityPipeline.PredefinedParameters.preliminary_multimobility.items()})
    def __init__(
        self,
        *,
        gait_sequence_detection: BaseGsdDetector,
        initial_contact_detection: BaseIcDetector,
        cadence_calculation: Optional[BaseCadDetector],
        stride_length_calculation: Optional[BaseSlDetector],
        walking_speed_calculation: Optional[BaseWsDetector],
        stride_selection: StrideFiltering,
        wba: WbAssembly,
        dmo_thresholds: Optional[pd.DataFrame],
        dmo_aggregation: AggregatorBase,
    ) -> None:
        super().__init__(
            gait_sequence_detection=gait_sequence_detection,
            initial_contact_detection=initial_contact_detection,
            cadence_calculation=cadence_calculation,
            stride_length_calculation=stride_length_calculation,
            walking_speed_calculation=walking_speed_calculation,
            stride_selection=stride_selection,
            wba=wba,
            dmo_thresholds=dmo_thresholds,
            dmo_aggregation=dmo_aggregation,
        )