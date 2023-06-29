# Manuscript work

For the work presented in [Beyond integration: modeling every pixel to obtain better structure factors from stills, IUCrJ Nov. 2020](https://doi.org/10.1107/S2052252520013007)


## Install CCTBX (optionally with CUDA support)

The below is for the NERSC GPU nodes, but it could easily be adapted to fit your local environment. 

### Build sources

Grab the bootstrap script and execute bootstrap with any modern python interpreter

```bash
# For GPU builds
module load cgpu gcc openmpi cuda # on NERSC only
# Verify nvcc is in your path
nvcc --version

mkdir ~/Crystal
cd ~/Crystal
wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
# if you arent doing a GPU build, then remove the config-flags argument
python bootstrap.py --builder=dials --use-conda --nproc=4 --config-flags="--enable_cuda" --python=38
```

This should create some sub-folders: 

* **~/Crystal/modules** (contains the sources), 
* **~/Crystal/build** (contains the setup script and build files), 
* **~/Crystal/mc3** (conatains the conda install, assuming no other conda was in the path, and/or no miniconda 
install folders were found in the home directory, ```source ~/Crystal/mc3/etc/profile.d/conda.sh```)
* **~/Crystal/conda_base** (contains the conda environment, can be activated using 
```conda activate ~/Crystal/conda_base```, but thats not necessary to use CCTBX or DIALS) 

### Test the build

```bash
# this sets up your environment to use CCTBX  and DIALS
source ~/Crystal/build/setpaths.sh
```

You can test the installation

```bash
mkdir ~/Crystal/test
cd ~/Crystal/test
libtbx.run_tests_parallel nproc=4 module=simtbx
```

### Install the repository for the manuscript work

Now grab the cxid9114 repo

```bash
cd ~/Crystal/modules # its important to place it in the modules folder
git clone https://github.com/dermen/cxid9114.git
# install git-lfs (if on nersc, just load the module
cd ~/Crystal/modules/cxid9114
module load git-lfs
cd ~/Crystal/modules/cxid9114
git lfs install
git lfs fetch
git lfs pull # this should bring some extra file content needed for the simulations

# this might not be necessary anymore but just in case:
cd ~/Crystal/modules 
git clone https://github.com/dermen/tilt_fit.git
```

### Adding some extra python modules

```bash
libtbx.python -m pip install hdf5plugin. # might be needed to view images
litbx.python -m pip install pandas jupyter
libtbx.refresh
libtbx.ipython # launch an interactive python shell
```

### Install the image format

Multi-panel images simulated with nanoBragg are saved in a custom-written format (```simtbx.nanoBragg.utils.H5AttributeGeomWriter```). The format is simple: the images are stored as 3D hdf5 datasets, and the dxtbx detector and beam models are converted to json strings and stored in the hdf5 dataset attribute field. The format reader can be installed as follows:

```bash
cd ~/Crystal/modules/cxid9114/format
dxtbx.install_format  -u FormatHDF5AttributeGeometry.py
```

### Install and test mpi4py

```bash
# Assuming mpicc is in your path (brought in on NERSC with the openmpi module shown above)
CC=gcc MPICC=mpicc libtbx.python -m pip install -v --no-binary mpi4py mpi4py
```

Test the mpi4py build by running the script below

```python
# store this in a script called ~/test_mpi.py
from mpi4py import MPI
import socket
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print(mpi_rank, mpi_size, socket.gethostname())
```

using mpirun (or srun on a nersc compute node)

```bash
# Test multi-node mpi 
srun -N2 -n2 -c2 libtbx.python ~/test_mpi.py

# or 
mpirun -n 6 libtbx.python ~/test_mpi.py
# The latter of which produced the following output on a Windows 10 PC running WSL version 2:

# 0 6 DESKTOP-FV2HS9K
# 2 6 DESKTOP-FV2HS9K
# 3 6 DESKTOP-FV2HS9K
# 4 6 DESKTOP-FV2HS9K
# 1 6 DESKTOP-FV2HS9K
# 5 6 DESKTOP-FV2HS9K
```


# Simulating the images

### Make a background image

```bash
cd ~/Crystal/modules/cxid9114/sim
libtbx.python d9114_mpi_sims.py  -odir . --bg-name mybackground.h5 --make-background   --sad 
```

You can view the background after installing the necessary dxtbx format class (as shown above)

```bash
dials.image_viewer mybackground.h5
```

### Make the diffracton patterns

Below is a script that can run on a PC to generate 2 diffraction images:

```bash
libtbx.python d9114_mpi_sims.py  -o test -odir some_images --add-bg --add-noise --profile gauss --bg-name mybackground.h5 -trials 2  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 1 --mos_spread_deg 0.01  --saveh5 --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115
```

To generate the two images (indicated by ```-trials 2```) it took 2min 40sec on a macbook. The images simulated for 
the paper included ```--mos_doms 50```, so expect that to take 100x longer to simulate (2x per mosaic block, as mosaic 
blocks are simulated in pairs to form an even distribution, such that the average mosaic missetting angle is 0 deg). We 
generated all images for the paper on a GPU node at the NERSC supercomputer. If you built CCTBX with cuda enabled, then 
you can also run the GPU simulation by adding the arguments ```-g 1  --gpu```, where ```-g``` specifies the number of 
GPU devices on the compute node. This will make simulating the images much faster. Also, the script can be run using 
MPI, parallelizing over images. The full command used at NERSC (1 compute node with 20 MPI ranks utilizing 8 GPUs) was 

```bash
srun -n 20 -c 2 libtbx.python d9114_mpi_sims.py  -o test -odir some_images --add-bg --add-noise --profile gauss --bg-name mybackground.h5 -trials 25  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 50 --mos_spread_deg 0.01  --saveh5 --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115 -g 8 --gpu
```

Note the above command has `--trials 25` and is run with 20 mpi tasks (`srun -n 20`), hence it will produce a total of 500 simulated shots. MPI Rank 0 will monitor GPU usage by periodically printing the result of nvidia-smi to the screen.

The simulated images can be opened with the DIALS image viewer. Note, the spots are large - this is because we 
intentionally simulated small mosaic domains in order to reduce aliasing errors that arise for Bragg spots that are much smaller than the solid angle subtended by a pixel. In such a case, one needs to increase the oversample factor to ensure proper sampling within each pixel, however this adds to the overall computational time (increasing the oversample from e.g. 1 to 5 incurs a 125x [!] increase in run-time). 


# Process the images with DIALS

## Indexing and integration

We will index and integrate the images using ```dials.stills_process```. Make a text file containing the simulated
image filenames and pass it along to stills_process 

```
find /path/to/some_images -name "test*h5" > img_files.txt
dials.stills_process process.phil  file_list=img_files.txt mp.nproc=5 output.output_dir=indexed
```

where process.phil is a file that contains the following lines of text

```
spotfinder.threshold.algorithm=dispersion
spotfinder.threshold.dispersion.gain=28
spotfinder.threshold.dispersion.global_threshold=40
spotfinder.threshold.dispersion.kernel_size=[2,2]
spotfinder.threshold.dispersion.sigma_strong=1
spotfinder.threshold.dispersion.sigma_background=6
spotfinder.filter.min_spot_size=2

indexing.method=fft1d
indexing.known_symmetry.unit_cell=79.1,79.1,38.4,90,90,90
indexing.known_symmetry.space_group=P43212
indexing.stills.set_domain_size_ang_value=500

integration.summation.detector_gain=28
```

Indexing will output reflection files and experiment files. For a detailed discription of these files, see the 
references by Brewster linked to in the manuscript. Here, we use the parameter ```set_domain_size_ang_value``` 
in order to intentionally over-predict the diffraction pattern. These over-predictions will provide more information 
for the diffBragg refinement. 

## Merge the data

To merge the data, run the command

```
mkdir merged # make an output folder for merge results
srun -n5 -c2 cctbx.xfel.merge merge.phil input.path=indexed output.output_dir=merged
```

where ```merge.phil``` is a text file containing the merging parameters:

```
input.parallel_file_load.method=uniform
filter.algorithm=unit_cell
filter.unit_cell.value.target_unit_cell=79.1,79.1,38.4,90,90,90
filter.unit_cell.value.target_space_group=P43212
filter.unit_cell.value.relative_length_tolerance=0.02
filter.outlier.min_corr=-1.0
select.algorithm=significance_filter
scaling.unit_cell=79.1,79.1,38.4,90,90,90
scaling.space_group=P43212
scaling.algorithm=mark1
scaling.resolution_scalar=0.96
postrefinement.enable=False
postrefinement.algorithm=rs
merging.d_min=2
merging.merge_anomalous=False
merging.set_average_unit_cell=True
merging.error.model=errors_from_sample_residuals
statistics.n_bins=10
output.do_timing=True
output.log_level=1
```

The merge result is now in ```merged/iobs_all.mtz```, and the log file containing merge statistics is 
```~/Crystal/merge/iobs_main.log```.

# Process the images with diffBragg


## Stage 1: Per-shot refinement with hopper

In the previous step we created a bunch of experiment/reflection files using stills_process. Now we can input these 
files to the diffBragg refinement program `hopper`. We will use integration files for this trial. The integration files designate
regions of interest on the diffraction images corresponding to where Bragg scattering is expected to be observed. We
purposely over-predicted when running stills_process to ensure we use a large fraction of the pixels for diffBragg

Begin by forming a two-column file containing experiments in the first column, and the corresponding reflection files in 
the second column. If you list all the ```integrated``` files in the ```indexed``` folder you will notice for each 
successfuly indexed image there is an *integrated.refl file and a *integrated.expt file. Together these should form a row 
in the txt file. You can create the text file by running the simple python script from the appropriate directory:

```
diffBragg.make_input_file  indexed/ integ_exp_ref.txt --splitDir splits --exptSuffix integrated.expt --reflSuffix integrated.refl
``` 

With this txt file we can then run stage_one refinement: 

```
DIFFBRAGG_USE_CUDA=1 srun -N4 --tasks-per-node=4 --cpus-per-gpu=1 --gpus-per-node=4 hopper hopper.phil exp_ref_spec_file=integ_exp_ref.txt num_devices=4 outdir=stage1
```

where the phil file contains the following parameters

```
roi {
  shoebox_size = 15
  fit_tilt = True
  reject_edge_reflections = False
  pad_shoebox_for_background_estimation=0
}

fix {
  detz_shift = True
  ucell=False
  Nabc=False
  G=False
  RotXYZ=False
}

sigmas {
  ucell = .1 .1
  RotXYZ = .01 .01 .01
  G = 1
  Nabc = .1 .1 .1
}

init {
  Nabc = 15 15 15
  G = 1e5
}

refiner {
  verbose = 0
  sigma_r = 3
  adu_per_photon = 28
}

simulator {
  crystal.has_isotropic_ncells = True
  structure_factors.mtz_name = merged/iobs_all.mtz 
  structure_factors.mtz_column = "Iobs(+),SIGIobs(+),Iobs(-),SIGIobs(-)"
  beam.size_mm = 0.001
}
```

Note, the file ```iobs_all.mtz``` is the merge output from running ```cctbx.xfel.merge```, as shown above. It is the 
initial guess of the structure factors.  The parameter descriptions can be viewed by executing

```
simtbx.diffBragg.stage_one -c -a2
```

## New predictions from the optimized models

Before running structure factor refinement, we use the optimized stage_one diffBragg models in order to define
new regions of interest on the camera. These regions of interest contain the pixels that will be used in structure 
factor refinement. This is a general prediction method with respect to the incident spectra and mosaicity (though in this
example we assumed a monochromatic spectrum). We use an oversample=1 override to speed up the computation, as well as an 
Ncells_abc override in order to overpredict reflections. Over-prediction is important for diffBragg, because it provides
a means for the optimizer to penalize itself if it models intensity in regions where there is no observed scattering.

TO run `diffBragg.integrate` first create a dummie file. This file is for the `predictions {}` phil parameters (see `diffBragg/phil.py` for full the list), however here we will simply pass them from the command line using the `--cmdlinePhil` flag. We also need to provide the phil file used for stills_process because we need the spot finding and integration parameters. (Note, if you see a message indicating `'Integration failed'`, it doesn't matter, all that matters is the `predicted.refl` files that `diffBragg.integrate` writes):

```
# create an empty dummie file
touch pred.phil

# now, run predictions:
DIFFBRAGG_USE_CUDA=1 srun -N4 --tasks-per-node=4 --cpus-per-gpu=1 --gpus-per-node=4 diffBragg.integrate  pred.phil process.phil stage1 stage1/predict --cmdlinePhil oversample_override=1 Nabc_override=[7,7,7] resolution_range=[2,100]  threshold=1 label_weak_col=rlp --numdev 4
```

The final stdout will be 

```
Reflections written to folder stage1/predict.

Wrote /global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalNew/modules/cxid9114/sim/stage1/predict/preds_for_hopper.pkl (best_pickle option for simtbx.diffBragg.hopper) and /global/cfs/cdirs/lcls/dermen/d9114_sims/CrystalNew/modules/cxid9114/sim/stage1/predict/preds_for_hopper.txt (exp_ref_spec option for simtbx.diffBragg.hopper). Use them to run the predictions through hopper. Use the centroid=cal option to specify the predictions
```

The output pandas dataframe pickle is a suitable input for stage_two (global) diffBragg refinement. It contains the data on the optimized experiments and the corresponding diffBragg prediction reflection tables.


## Stage 2: Structure factor refinement

Using the output file from the predictions script we can run structure factor refinement. In this case, we know the ground
truth structure factors, therefore we can use them as a reference to observe how accurate the optimization is with each 
iteration. The following python script can be used to generate the ground truth structure factor MTZ file. The PDB file
should be in the sim folder of this repository, however it can be downloaded from the PDB using the command 
```iotbx.fetch_pdb 4bs7```. Also, the file `scanned_fp_fdp.tsv` represents the anomlous contribution to the scattering, and is 
found in the ```sf``` folder of this repository: 

```python
from cxid9114.sf import struct_fact_special
from cxid9114.parameters import WAVELEN_HIGH
Famp = struct_fact_special.sfgen(WAVELEN_HIGH, "cxid9114/sim/4bs7.pdb", dmin=1.9, yb_scatter_name="cxid9114/sf/scanned_fp_fdp.tsv")
Famp = Famp.as_amplitude_array()
mtz = Famp.as_mtz_dataset(column_root_label="F").mtz_object()
mtz.write("cxid9114_grndtruth.mtz")
```

```
DIFFBRAGG_USE_CUDA=1 srun -N4 --tasks-per-node=4 --cpus-per-gpu=1 --gpus-per-node=4 simtbx.diffBragg.stage_two stage_two.phil io.output_dir=stage2 pandas_table=stage1/predict/preds_for_hopper.pkl
```

where ```stage_two.phil``` contains the following

```
roi {
  shoebox_size = 15
  fit_tilt = True
  reject_edge_reflections = False
  fit_tilt_using_weights = False
  pad_shoebox_for_background_estimation=0
}

sigmas {
  G = 1
  Fhkl = 1
}

refiner {
  refine_Fcell = [1]
  refine_spot_scale = [1]
  max_calls = [450]
  #sensitivity.spot_scale = 1
  #sensitivity.fcell = 1
  ncells_mask = 111
  tradeps = 1e-20
  verbose = 0
  sigma_r = 3
  adu_per_photon = 28
}

simulator {
  crystal.has_isotropic_ncells = True
  structure_factors.mtz_name = merged/iobs_all.mtz 
  structure_factors.mtz_column = "Iobs(+),SIGIobs(+),Iobs(-),SIGIobs(-)"
  beam.size_mm = 0.001
}
```

The output folder ```stage2``` contains information necssary to construct the optimized miller array

```python
import numpy as np
from cctbx.array_family import flex
from cctbx import miller
from cctbx.crystal import symmetry

# load the asu mapping and the last iteration fcell file from the stage_two output folder:

asu_map = np.load("stage2/f_asu_map.npy", allow_pickle=True)[()]

for i in 0, 450:

  Famp = np.load("stage2/_fcell_trial0_iter%d.npz" % i)["fvals"]

  # construct a symmetry object for this lysozyme crystal:
  sym = symmetry((79.1,79.1,38.4,90,90,90), "P43212")
  val_at_index = {hkl: Famp[i] for i,hkl in asu_map.items()}
  hkl_list = list(asu_map.values())
  # construct a miller array object and save to mtz
  millset = miller.set(sym, flex.miller_index(hkl_list), anomalous_flag=True)
  mill_ary = millset.array(data=flex.double([val_at_index[hkl] for hkl in hkl_list]  ))
  print(mill_ary.completeness())  # shows the data completeness
  mtz = mill_ary.as_mtz_dataset("Famp").mtz_object()
  mtz.write("stage2/optimized_%d.mtz" % i)
```

