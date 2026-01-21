#!/bin/bash

START_MERGE=$(date +"%s")

TRIAL=$1

#cctbx
#source /build/setpaths.sh
MERGE_ROOT=${HOME}/MERGE

# output directory
OUT_DIR=${HOME}/MERGE

# trial
TRIAL_F="$(printf "%03d" ${TRIAL})"

# setup playground

mkdir -p ${OUT_DIR}/${TRIAL_F}/out
mkdir -p ${OUT_DIR}/${TRIAL_F}/stdout
mkdir -p ${OUT_DIR}/${TRIAL_F}/tmp

#scaling.model=${MERGE_ROOT}/4bs7.pdb \
#statistics.cciso.mtz_file=${MERGE_ROOT}/4ngz.mtz \
#statistics.cciso.mtz_column_F=Iobs(+) \

export effective_params="\

input.path=$2 \

input.parallel_file_load.method=uniform \

filter.algorithm=unit_cell \

filter.unit_cell.value.target_unit_cell=79.1,79.1,38.4,90,90,90 \

filter.unit_cell.value.target_space_group=P43212 \

filter.unit_cell.value.relative_length_tolerance=0.02 \

filter.outlier.min_corr=-1.0 \

select.algorithm=significance_filter \

scaling.unit_cell=79.1,79.1,38.4,90,90,90 \

scaling.space_group=P43212 \

scaling.algorithm=mark0 \

scaling.model=$3 \

scaling.mtz.mtz_column_F=Iobs(+) \

scaling.mark0.fit_reference_to_experiment=True \

scaling.mark0.fit_offset=False \

scaling.resolution_scalar=0.96 \

postrefinement.enable=True \

postrefinement.algorithm=rs \

merging.d_min=1.9 \

merging.merge_anomalous=False \

merging.set_average_unit_cell=True \

merging.error.model=errors_from_sample_residuals \

statistics.n_bins=10 \

output.prefix=${TRIAL_F} \

output.output_dir=${OUT_DIR}/${TRIAL_F}/out 

output.tmp_dir=${OUT_DIR}/${TRIAL_F}/tmp \

output.do_timing=True \

output.log_level=1"

#run merge

cctbx.xfel.merge ${effective_params}

END_MERGE=$(date +"%s")

ELAPSED=$((END_MERGE-START_MERGE))

echo TotalElapsed_OneCore ${ELAPSED} ${START_MERGE} ${END_MERGE}
