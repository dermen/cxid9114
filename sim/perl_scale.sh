#!/bin/bash

#SBATCH -N 30
#SBATCH --ntasks-per-node=12
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=3
#SBATCH -t 12
#SBATCH -q regular 
#SBATCH -C gpu
#SBATCH -A m4326_g
#SBATCH -J dfbrg 
#SBATCH --error=%J.err
#SBATCH --output=%J.out

ODIR=out.${SLURM_JOBID}
DIFFBRAGG_USE_CUDA=1
SRUN="srun -N30 --ntasks-per-node=12 --gpus-per-node=4 --cpus-per-gpu=3 -c2"

# <><><><><
# SIMULATE
# <><><><><
# 1 process background simulation (writes mybackground.h5)
libtbx.python d9114_mpi_sims.py  -odir $ODIR --bg-name mybackground.h5 --make-background  --sad 

# N process full image simulations (writes 10 images per rank in the some_images folder)
$SRUN libtbx.python d9114_mpi_sims.py  -o test -odir ${ODIR}/some_images --add-bg --add-noise --profile gauss --bg-name ${ODIR}/mybackground.h5 -trials 20  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 50 --mos_spread_deg 0.01  --saveh5 --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115 -g 4 --gpu

sleep 30

# <><><><><><><><><><><><
# CONVENTIONAL PROCESSING
# <><><><><><><><><><><><

# creates an input file for stills process
find ${ODIR}/some_images/ -name "test*h5" > ${ODIR}/images.txt

# run indexing and integration
$SRUN dials.stills_process process.phil  file_list=${ODIR}/images.txt mp.method=mpi output.output_dir=${ODIR}/indexed logging_option=disabled

sleep 30

# run merging
$SRUN cctbx.xfel.merge merge.phil input.path=${ODIR}/indexed output.output_dir=${ODIR}/merged

sleep 30

# <><><><><>
# DIFFBRAGG 
# <><><><><>

# create an input file for diffBragg (creates integ_exp_ref.txt and splits folder)
$SRUN diffBragg.make_input_file  ${ODIR}/indexed/ ${ODIR}/integ_exp_ref.txt --splitDir ${ODIR}/splits --exptSuffix integrated.expt --reflSuffix integrated.refl

sleep 30

# run per-shot refinement using hopper 
$SRUN  hopper hopper.phil exp_ref_spec_file=${ODIR}/integ_exp_ref.txt num_devices=4 outdir=${ODIR}/stage1 simulator.structure_factors.mtz_name=${ODIR}/merged/iobs_all.mtz

sleep 30

# draw new prediction shoeboxes from the hopper results
$SRUN diffBragg.integrate  pred.phil process.phil ${ODIR}/stage1 ${ODIR}/stage1/predict --cmdlinePhil oversample_override=1 Nabc_override=[7,7,7] resolution_range=[2,100]  threshold=1 label_weak_col=rlp --numdev 4

sleep 30

# run stage 2 ensemble refinement
$SRUN simtbx.diffBragg.stage_two stage_two.phil io.output_dir=${ODIR}/stage2 pandas_table=${ODIR}/stage1/predict/preds_for_hopper.pkl num_devices=4 simulator.structure_factors.mtz_name=${ODIR}/merged/iobs_all.mtz


#DIFFBRAGG_USE_CUDA=1 srun -N1 --ntasks-per-node=32 --cpus-per-gpu=8 --gpus-per-node=4 -c2 -n32   ens.hopper  watchmans_ease/hopper13_fall23/filtered2.pkl hopper13_fall23.phil --outdir test.ens --maxSigma 10 --saveFreq 10 --cmdlinePhil max_process=128 fix.Fhkl=False fix.G=False fix.RotXYZ=True fix.Nabc=True fix.ucell=True sigmas.G=0.1 --refl predictions


DIFFBRAGG_USE_CUDA=1 srun -N4 --ntasks-per-node=32 -c2 --gpus-per-node=4 --cpus-per-gpu=8 -n16 ens.hopper  watchmans_ease/hopper13_fall23/33perc_filt.pkl hopper13_fall23.phil --outdir watchmans_ease/hopper13_fall23/33perc_Nref --saveFreq 10 --cmdlinePhil max_process=512 fix.Fhkl=False fix.G=False fix.RotXYZ=True fix.Nabc=False fix.ucell=True  sigmas.G=0.1 fix.diffuse_sigma=True fix.diffuse_gamma=True init.diffuse_sigma=[.12,.11,.1] init.diffuse_gamma=[1e3,1e3,1e3] diffuse_stencil_size=1 sigmas.diffuse_gamma=[.7,.7,.7] sigmas.diffuse_sigma=[.5,.5,.5] diffuse_orientation=1 use_diffuse_models=False logfiles=True  --refl predictions --saveAll

DIFFBRAGG_USE_CUDA=1 srun -N4 --ntasks-per-node=32 -c2 --gpus-per-node=4 --cpus-per-gpu=8 -n16 ens.hopper  watchmans_ease/hopper13_fall23/33perc_filt.pkl hopper13_fall23.phil --outdir watchmans_ease/hopper13_fall23/33perc_Dref --saveFreq 10 --cmdlinePhil max_process=512 fix.Fhkl=False fix.G=False fix.RotXYZ=True fix.Nabc=True fix.ucell=True  sigmas.G=0.1 fix.diffuse_sigma=False fix.diffuse_gamma=False init.diffuse_sigma=[.12,.11,.1] init.diffuse_gamma=[1e3,1e3,1e3] diffuse_stencil_size=1 sigmas.diffuse_gamma=[.7,.7,.7] sigmas.diffuse_sigma=[.5,.5,.5] diffuse_orientation=1 use_diffuse_models=True logfiles=True  --refl predictions --saveAll
