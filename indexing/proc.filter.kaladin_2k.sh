#!/bin/bash

#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH -c 80
#SBATCH --account=m1759

module load cuda
module load mvapich2
source $HOME/Crystal/build/setpaths.sh

# $1 is the directory basename (not full path) containing stills process files
export IDXDIR=$DD/kaladin_2k
export AGGDIR=$DD/agg_kaladin_2k
dials.stills_process params.phil $HOME/kaladin_2_imgs/job*/*.h5.npz mp.nproc=40 output.output_dir=$IDXDIR

cd $ODIR
dials.combine_experiments *refined* *indexed* reference_from_experiment.detector=0
cctbx.xfel.filter_experiments_by_rmsd combined.*

cd $HOME/cxid9114/indexing
srun -n 40 -c 2  libtbx.python process_filtered_mpi.py -o $AGGDIR --Z 8 --dilate 2 --sbpad 3 --thresh 1e-3 --ngpu 8 --filteredexpt $IDXDIR/filtered.expt --defaultF 1e3 

cd $HOME/cxid9114/io
srun -n 40 -c 2  libtbx.python filter_bboxes.py --glob "$AGGDIR/process_rank*h5" --gain 28 --snrmin 0.2 --onboundary --tilterrmax 250  --reslow 999 --reshigh 2.1
