"""
.. step_by_step_pipeline:

Step-by-Step Breakdown of a pipeline example
===============================================

This example shows how to build a full analysis pipeline using multigait algorithms and provides all the individual steps.

"""

# %%
# Load example data
# -----------------

import pandas as pd
from multigait.utils.interp import map_seconds_to_regions
from multigait.pipeline.utils.datapoint_check import check_gait_datapoint_completeness
from examples.example_data.example_constructor import construct_datapoint_from_files

data = construct_datapoint_from_files()
imu_data = data.data_ss
sampling_rate_hz = data.sampling_rate_hz

# Checking if the datapoint is complete (if incomplete pipeline might work or fail depending on the missing aspects)
is_complete = check_gait_datapoint_completeness(data)
if not is_complete:
    print("Warning: Datapoint is incomplete!")

# %%
# Step 1: GSD
# -------------------------------
from multigait.GSD.GSD3 import KheirkhahanGSD

gsd = KheirkhahanGSD(version="improved_lowback").detect(imu_data)

gait_sequences = gsd.gs_list_
gait_sequences

# %%
# Starting from here, all the processing will happen per gait sequence.
# We will go through the steps just for a single gait sequence first and later put everything in a loop.
first_gait_sequence = gait_sequences.iloc[0]
first_gait_sequence_data = imu_data.iloc[
    first_gait_sequence.start : first_gait_sequence.end
]


# %%
# Step 2: Initial Contact Detection
# ---------------------------------

from multigait.ICD.ICD2 import McCamleyIC

icd = McCamleyIC(version="improved_lowback").detect(
    first_gait_sequence_data, sampling_rate_hz=100
)
ic_list = icd.ic_list_
ic_list


# %%
# Step 3: Cadence Calculation
# ---------------------------
from multigait.CAD.cad import Cadence

cad = Cadence()
cad.calculate(
    first_gait_sequence_data,
    initial_contacts=ic_list,
    sampling_rate_hz=100,
)

cad_per_sec = cad.cadence_per_sec_
cad_per_sec

# %%
# Step 4: Stride Length Calculation
# ---------------------------------
from multigait.SL.SL1 import WeinbergSL

sl = WeinbergSL(version="wrist").calculate(
    data=first_gait_sequence_data, initial_contacts=ic_list, sampling_rate_hz=100
)

sl_per_sec = sl.stride_length_per_sec_
sl_per_sec

# %%
# Step 5: Walking Speed Calculation
# ---------------------------------

from multigait.WS.walking_speed import Ws

ws = Ws().calculate(cadence_per_sec=cad_per_sec, stride_length_per_sec=sl_per_sec)

ws_per_sec = ws.walking_speed_per_sec_
ws_per_sec


# %%
# After going through the steps for a single gait sequence,
# we would then put all the data together to calculate the final results per WB.
# But let's first put all the processing into an easy-to-read loop.
#
# Actual Pipeline
# ---------------
# We first define all the algorithms we want to use.
from multigait.CAD.cad import Cadence
from multigait.GSD.GSD3 import KheirkhahanGSD
from multigait.ICD.ICD2 import McCamleyIC
from multigait.SL.SL1 import WeinbergSL
from multigait.WS.walking_speed import Ws

gsd = KheirkhahanGSD(version="improved_lowback")
icd = McCamleyIC(version="improved_lowback")
cad = Cadence()
sl = WeinbergSL()
speed = Ws()

# %%
# Then we calculate the gait sequences as previously.
#
# Note that some of the algorithms might need the participant metadata.
# Hence, we pass it as keyword argument to all the algorithms.
gsd.detect(imu_data)
gait_sequences = gsd.gs_list_

# %%
# Then we use a nested iterator to go through all the gait sequences and process them.
# To learn more about this iterator,
# check out the example about the :ref:`Gait Sequence Iterator <gs_iterator_example>`.
# Note, that we use the special ``r`` object to store the results of each step and the ``subregion`` method to
# elegantly handle the refined gait sequence.
from multigait.pipeline.iterator import GsIterator

gs_iterator = GsIterator()

for (_, gs_data), r in gs_iterator.iterate(imu_data, gait_sequences):
    icd = icd.clone().detect(gs_data, sampling_rate_hz=sampling_rate_hz)
    r.ic_list = icd.ic_list_

    # cadence
    cad = cad.clone().calculate(
        gs_data,
        initial_contacts=icd.ic_list_,
        sampling_rate_hz=sampling_rate_hz,
    )
    r.cadence_per_sec = cad.cadence_per_sec_

    # stride length
    sl = sl.clone().calculate(
        gs_data,
        initial_contacts=icd.ic_list_,
        sampling_rate_hz=sampling_rate_hz,
    )
    r.stride_length_per_sec = sl.stride_length_per_sec_

    # walking speed
    speed = speed.clone().calculate(
        gs_data,
        initial_contacts=icd.ic_list_,
        cadence_per_sec=cad.cadence_per_sec_,
        stride_length_per_sec=sl.stride_length_per_sec_,
        sampling_rate_hz=sampling_rate_hz,
    )
    r.walking_speed_per_sec = speed.walking_speed_per_sec_

# %%
# Now we can access all accumulated and offset-corrected results from the iterator.
results = gs_iterator.results_

results.ic_list

# %%
# We combine all per-sec results into one.
combined_results = pd.concat(
    [
        results.cadence_per_sec,
        results.stride_length_per_sec,
        results.walking_speed_per_sec,
    ],
    axis=1,
)
combined_results

# %%
# Using the combined results, we want to define walking bouts.
# As walking bouts are defined based on strides, we need to turn the ICs into strides and
# the per-second values into per-stride values by using interpolation.
# We also calculate the stride duration here.
from multigait.pipeline.utils.ic_to_stride import strides_list_from_ic_list_no_lrc

stride_list = (
    results.ic_list.groupby("gs_id", group_keys=False)
    .apply(strides_list_from_ic_list_no_lrc)
    .assign(stride_duration_s=lambda df_: (df_.end - df_.start) / sampling_rate_hz)
)
stride_list

# %%
# This initial stride list is completely unfiltered, and might contain very long strides, in areas where initial
# contacts were not detected, or the participant was not walking for a short moment.
# The stride list will be filtered later as part of the WB assembly.
#
# For now, we are using linear interpolation to map the per-second cadence values to per-stride values and derive
# approximated stride parameters.
from multigait.pipeline.utils._operations import create_multi_groupby

stride_list_with_approx_paras = create_multi_groupby(
    stride_list,
    combined_results,
    "gs_id",
    group_keys=False,
).apply(map_seconds_to_regions, sampling_rate_hz=sampling_rate_hz)

stride_list_with_approx_paras

# %%
# Now the final strides are regrouped into walking bouts.
# For this we ignore which gait sequence the strides belong to, hence we remove the ``gs_id`` from the index, but keep
# it around as column for debugging.
from multigait.pipeline.utils._stride_filtering import StrideFiltering
from multigait.pipeline.utils._wb_assembly import WbAssembly

flat_index = pd.Index(
    ["_".join(str(e) for e in s_id) for s_id in stride_list_with_approx_paras.index],
    name="s_id",
)
stride_list_with_approx_paras = (
    stride_list_with_approx_paras.reset_index("gs_id")
    .rename(columns={"gs_id": "original_gs_id"})
    .set_index(flat_index)
)

# %%
# Then we apply the stride selection (note that we have additional rules in case the stride length is available) and
# then group the remaining strides into walking bouts.
ss = StrideFiltering().filter(
    stride_list_with_approx_paras, sampling_rate_hz=sampling_rate_hz
)

wba = WbAssembly().assemble(
    ss.filtered_stride_list_,
    raw_initial_contacts=results.ic_list,
    sampling_rate_hz=sampling_rate_hz,
)

final_strides = wba.annotated_stride_list_

# %%
# Here we calculate the additional DMOs for each WB
from multigait.pipeline.utils._var_dmos import within_wb_var

var_dmos = within_wb_var(final_strides)

# %%
# We also have meta information about the WBs available.
per_wb_params = wba.wb_meta_parameters_
per_wb_params.drop(columns="rule_obj").T

# %%
# We extend them further with the per-stride parameters.
# The per-stride parameters are aggregated per WB using their mean.
params_to_aggregate = [
    "stride_duration_s",
    "cadence_spm",
    "stride_length_m",
    "walking_speed_mps",
]
per_wb_params = pd.concat(
    [
        per_wb_params,
        final_strides.reindex(columns=params_to_aggregate).groupby(["wb_id"]).mean(),
    ],
    axis=1,
)

# Here we add the variability DMOs
per_wb_params = pd.concat([per_wb_params, var_dmos], axis=1)
per_wb_params.drop(columns="rule_obj").T

# %%
# Adding alpha calculation
from multigait.pipeline.utils.alpha import compute_alpha_mle

per_wb_params = compute_alpha_mle(per_wb_params)

# %%
from multigait.pipeline.utils._thresholds import get_thresholds, apply_thresholds

# load single min/max thresholds from:
thresholds = get_thresholds()

per_wb_params_mask = apply_thresholds(
    per_wb_params,
    thresholds=thresholds,
    height_m=data.participant_metadata.get("height_m"),
)
per_wb_params_mask.T

# %%
#
# We can see that we either get NaN (for parameters that are not checked) or True/False values for each parameter.
#
# This output together with the per-WB parameters would then normally be used in some aggregation step to calculate
# single values per participant, day, or other grouping criteria.
# Depending on the use-case, this aggregation can be performed withing the "per-recording" pipeline or as a separate
# step after processing all recordings.
#
# We add information on participant_id and measurement_date so we can use the GenericAggregator
#
# Here, we perform it per recording and calculate a single values from all the WBs.
from multigait.aggregation._generic_aggregator import GenericAggregator

agg = GenericAggregator(**GenericAggregator.PredefinedParameters.single_day)
agg_results = agg.aggregate(
    per_wb_params, wb_dmos_mask=per_wb_params_mask
).aggregated_data_
agg_results.T
print(agg_results)
print(agg_results.columns)


# %%
# Laboratory Aggregation example
# In this aggregation we only take into account all WBs together, without separating them into days or WB durations.
from multigait.aggregation._lab_aggregator import LaboratoryAggregator

agg = LaboratoryAggregator(**LaboratoryAggregator.PredefinedParameters.single_recording)
agg_results = agg.aggregate(
    per_wb_params, wb_dmos_mask=per_wb_params_mask
).aggregated_data_
agg_results.T
print(agg_results.T)
print(agg_results.columns)
