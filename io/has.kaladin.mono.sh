#!/bin/bash
#SBATCH --qos=premium
#SBATCH --time=08:00:00
#SBATCH --nodes=128
#SBATCH --constraint=haswell
#SBATCH --output=has.mono.out
#SBATCH --account=m3289

source $HOME/dermen.sh
source /global/common/software/lcls/Crystal/build/setpaths.sh

srun -n 2048 -c 2  libtbx.python ./global_bboxes_fcell.py --character syl3 --glob "$DD/kaladin/agg_1e6/process_rank*.h5" --gainval 28 --sad --Ncells_size 9.98  --oversample 0 --globalNcells --bs7real --Fobs ../sf/kaladin_merge.pkl --usepreoptAmat --usepreoptscale --globalNcells --unknownscale --optoutname $DD/kaladin/0-29_fcell_scales_too_refined.pkl --verbose --forcemono --outdir $DD/kaladin/0-29_fcell_scales_too --partition --partitiontime 50  --Fref ../sf/bs7_real.pkl --scaleR1  --fcell 1 --scale 1 --bg 1 --bmatrix 0 --umatrix 0 --ncells 1  | tee $DD/kaladin/0-29_fcell_scales_too.log

