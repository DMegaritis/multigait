# %%
# Load example data
# Loading an example datapoint with data from a wrist worn device

from src.multigait.pipeline.utils.datapoint_check import (
    check_gait_datapoint_completeness,
)
from examples.example_data.example_constructor import construct_datapoint_from_files

data = construct_datapoint_from_files()

# Checking if the datapoint is complete (if incomplete pipeline might work or fail depending on the missing aspects)
is_complete = check_gait_datapoint_completeness(data)
if not is_complete:
    print("Warning: Datapoint is incomplete!")
# %%
# Running as a single pipeline

from src.multigait.pipeline.utils._stride_filtering import StrideFiltering
from src.multigait.pipeline.utils._wb_assembly import WbAssembly
from src.multigait.pipeline.utils._thresholds import get_thresholds
from src.multigait.aggregation._generic_aggregator import GenericAggregator
from src.multigait import Ws
from src.multigait import WeinbergSL
from src.multigait.CAD.cad import Cadence
from src.multigait.ICD.ICD2 import McCamleyIC
from src.multigait import KheirkhahanGSD

gsd = KheirkhahanGSD(version="wrist")
icd = McCamleyIC()
ws = Ws()
sl = WeinbergSL()
cad = Cadence()
ss = StrideFiltering()
wba = WbAssembly()
thresholds = get_thresholds()
agg = GenericAggregator(**GenericAggregator.PredefinedParameters.single_day)

from src.multigait.pipeline.multimobility_pipeline import MultiGaitPipeline

pipeline = MultiGaitPipeline(
    gait_sequence_detection=gsd,
    initial_contact_detection=icd,
    initial_contact_detection_sl=None,
    cadence_calculation=cad,
    stride_length_calculation=sl,
    walking_speed_calculation=ws,
    stride_selection=ss,
    wba=wba,
    dmo_thresholds=thresholds,
    dmo_aggregation=agg,
)

pipeline.safe_run(data)

# %%
# The results are stored in the pipeline object.
# And basically all the individual results that are shown above are also available in the pipeline object.
#
# For example the per stride parameters:
pipeline.raw_per_stride_parameters_

# %%
# The per-wb parameters:
pipeline.per_wb_parameters_

# %%
# And the aggregated parameters:
pipeline.aggregated_parameters_

# %%
# Running the preliminary suggested pipeline separately.
# For this, we do not need to specify the algorithms to be used.

from src.multigait.pipeline.multimobility_pipeline import (
    MultiGaitPipelineMultimorbidityImpaired,
)

pipeline = MultiGaitPipelineMultimorbidityImpaired()

pipeline.safe_run(data)

# %%
# The results are stored in the pipeline object.
# And basically all the individual results that are shown above are also available in the pipeline object.
#
# For example the per stride parameters:
pipeline.raw_per_stride_parameters_

# %%
# The per-wb parameters:
pipeline.per_wb_parameters_

# %%
# And the aggregated parameters:
pipeline.aggregated_parameters_
