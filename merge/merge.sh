#!/bin/bash

START_MERGE=$(date +"%s")

TRIAL=$1

MERGE_ROOT=$2

# output directory
OUT_DIR=$2

# trial
TRIAL_F="$(printf "%03d" ${TRIAL})"

# setup playground
mkdir -p ${OUT_DIR}/${TRIAL_F}/out
mkdir -p ${OUT_DIR}/${TRIAL_F}/stdout
mkdir -p ${OUT_DIR}/${TRIAL_F}/tmp

export effective_params="\

input.path=$3 \

input.parallel_file_load.method=uniform \

filter.algorithm=unit_cell \

filter.unit_cell.value.target_unit_cell=79.1,79.1,38.4,90,90,90 \

filter.unit_cell.value.target_space_group=P43212 \

filter.unit_cell.value.relative_length_tolerance=0.02 \

filter.outlier.min_corr=-1.0 \

select.algorithm=significance_filter \

scaling.unit_cell=79.1,79.1,38.4,90,90,90 \

scaling.space_group=P43212 \

scaling.algorithm=mark1 \

scaling.resolution_scalar=0.96 \

postrefinement.enable=False \

postrefinement.algorithm=rs \

merging.d_min=2 \

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

srun -n 40 -c 2 cctbx.xfel.merge ${effective_params}

END_MERGE=$(date +"%s")

ELAPSED=$((END_MERGE-START_MERGE))

echo TotalElapsed_OneCore ${ELAPSED} ${START_MERGE} ${END_MERGE}
