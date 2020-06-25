#!/bin/bash

export D91=$HOME/cxid9114
export DUMP=$DD/test_520
# SIMULATE

cd $D91/sim
srun -n 1 -c 2 libtbx.python ./d9114_mpi_sims.py  -o test_520 -odir $DUMP --add-bg --add-noise --profile gauss --bg-name test_520_background.h5 --make-background --cspad  -trials 1  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 1 --mos_spread_deg 0 --savenpz --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115 --savenoiseless --forcemono --savereadoutless

time srun -n 40 -c 2 ./d9114_mpi_sims.py  -o test_520 -odir $DUMP  --add-bg --add-noise --profile gauss --bg-name test_520_background.h5 --cspad  -trials 13  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 1 --mos_spread_deg 0 --savenpz --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115 --savenoiseless --forcemono --savereadoutless


cd $D91/indexing
dials.stills_process tutorial.phil $DUMP/job*/*.h5.npz mp.nproc=40 output.output_dir=$DUMP/index

cd $D91/merge
./merge.sh  1 $DUMP/merge  $DUMP/index/


cd $D91/sf
libtbx.python Fmtz_to_Fpkl.py  --input $DUMP/merge/001/out/001_all.mtz --output test_520_cpu_mark1.pkl


cd $DUMP/index
dials.combine_experiments *refined.expt *indexed.refl reference_from_experiment.detector=0
cctbx.xfel.filter_experiments_by_rmsd combined.*


cd $D91/indexing
srun -n40 -c2 libtbx.python process_filtered_mpi.py -o $DUMP/agg --Z 8  --dilate 2  --thresh 1e-3 --filteredexpt $DUMP/index/filtered.expt  --defaultF 1e3 --nocuda

cd $D91/io
srun -n40 -c2 libtbx.python filter_bboxes.py --glob "$DUMP/agg/*.h5" --gain 28  --snrmin 3 --onboundary  --reslow 999 --reshigh 5 --keeperstag stage1 --resoinfile

srun -n40 -c2 libtbx.python filter_bboxes.py --glob "$DUMP/agg/*.h5" --gain 28  --snrmin 0.2 --onboundary  --reslow 999 --reshigh 2.1 --keeperstag stage2 --resoinfile

# libtbx.python estimate_mosaic_parameter_m.py --expfile $DUMP/index/filtered.expt 

libtbx.python stage1_embarrassing.py --aggdir $DUMP/agg/ --mergepkl ../sf/test_520_cpu_mark1.pkl --outdir $DUMP/stage1 --initNcells 14.58

libtbx.python make_df2.py --filter --stage1dir $DUMP/stage1

cd $D91/indexing
libtbx.python make_new_experiment.py  --pkl $DUMP/stage1/df2.pkl  --exp $DUMP/index/filtered.expt  --tag $DUMP/index/after_stage1

time srun -n40 -c2 libtbx.python process_filtered_mpi.py -o $DUMP/agg_after_stage1 --Z 8  --dilate 2  --thresh 1e-3 --filteredexpt $DUMP/index/after_stage1.expt  --defaultF 1e3 --nocuda

cd $D91/io
srun -n40 -c2 libtbx.python filter_bboxes.py --glob "$DUMP/agg_after_stage1/*.h5" --gain 28  --snrmin 0.2 --onboundary  --reslow 999 --reshigh 2.1 --keeperstag stage2 --resoinfile

time srun -n 40 -c 2 libtbx.python fat_data.py --glob "$DUMP/agg_after_stage1/process*h5" --gainval 28 --sad --Ncells_size 9.927296 --globalNcells --oversample 0 --bs7real --Fobs ../sf/test_520_cpu_mark1.pkl --unknownscale 17789.77 --verbose --forcemono --scale 1 --ncells 0 --bg 0 --umatrix 0 --bmatrix 0 --fcell 1 --maxcalls 500 --keeperstags stage2 --Fref ../sf/bs7_real.pkl --spotscalesigma 1 --fcellsigma 1 --minmulti 1 --protocol global --xrefinedonly --NoRescaleFcellRes --outdir $DUMP/stage2 | tee $DUMP/stage2.log

libtbx.python check_fcell_convergence.py  --truthpkl ../sf/bs7_real.pkl  --datdir $DUMP/stage2 --trial 0 --stride 10 --mtzoutput test_520_after_stage2.mtz

cd $D91/sf
libtbx.python Fmtz_to_Fpkl.py --input ../io/test_520_after_stage2.mtz  --output test_520_after_stage2.pkl
