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
srun -n 20 -c 2 libtbx.python d9114_mpi_sims.py  -o test -odir some_images --add-bg --add-noise --profile gauss --bg-name mybackground.h5 -trials 2000  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 50 --mos_spread_deg 0.01  --saveh5 --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115 -g 8 --gpu
```

MPI Rank 0 will monitor GPU usage by periodically printing the result of nvidia-smi to the screen.

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

The program diffBragg is currently part of a feature branch of CCTBX. It is actively under development, and ultimately 
it will be merged into the master branch of CCTBX. For now, if you wish to use diffBragg, you must checkout the feature 
branch (this instructional will be updated once diffBragg is moved into CCTBX master):

```python
cd ~/Crystal/modules/cctbx_project
git checkout diffBragg
cd ~/Crystal/build
make

# go to an empty test folder and run the tests
cd ~/Crystal/tests
libtbx.run_tests_parallel module=simtbx nproc=5
```

If you configured CCTBX to use enable CUDA *as shown above* then you can test the GPU functionality. There is a single 
environment variable to turn this on, see below.

## Process with simtbx.diffBragg.stage_one

In the previous step we created a bunch of experiment/reflection files using stills_process. Now we can input these 
files to the diffBragg refinement program. We will use integration files for this trial. The integration files designate
regions of interest on the diffraction images corresponding to where Bragg scattering is expected to be observed. We
purposely over-predicted when running stills_process to ensure we use a large fraction of the pixels for diffBragg

Begin by forming a two-column file containing experiments in the first column, and the corresponding reflection files in 
the second column. If you list all the ```intrgrated``` files in the ```indexed``` folder you will notice for each 
successfuly indexed image there is a *integrated.refl file and a *integrated.expt file. Together these should form a row 
in the txt file. You can create the text file by running the simple python script from the appropriate directory:

```python
import glob
fnames = glob.glob("indexed/*integrated.expt")
o = open("exper_refls.txt", "w")
for exper in fnames:
    refl = exper.replace(".expt", ".refl")
    o.write("%s %s\n" % (exper, refl))
o.close()
``` 

With this txt file we can then run stage_one refinement: 

```
OMP_NUM_THREADS=2 srun -n 40 -c 2 simtbx.diffBragg.stage_one exper_refls_file=exper_refls.txt stage_one.phil  output.directory=stage_one usempi=True 
```

where the phil file contains the following parameters

```
roi.shoebox_size = 15
roi.fit_tilt = True
roi.reject_edge_reflections = False
roi.pad_shoebox_for_background_estimation=0
refiner.refine_Umatrix = [0,1]
refiner.refine_ncells = [1,0]
refiner.refine_spot_scale = [1,0]
refiner.refine_Bmatrix = [0,1]
refiner.max_calls = [100, 100]
refiner.sensitivity.unitcell = [.1, .1, .1, .1, .1, .1]
refiner.sensitivity.rotXYZ = [.01, .01, .01]
refiner.sensitivity.spot_scale = 1
refiner.sensitivity.ncells_abc = [.1,.1, .1]
refiner.ncells_mask = 111
refiner.tradeps = 1e-20
refiner.verbose = 0
refiner.sigma_r = 3
refiner.adu_per_photon = 28
simulator.crystal.has_isotropic_ncells = True
simulator.structure_factors.mtz_name = iobs_all.mtz 
simulator.structure_factors.mtz_column = "Iobs(+),SIGIobs(+),Iobs(-),SIGIobs(-)"
simulator.crystal.ncells_abc = 15,15,15
simulator.init_scale = 1e5
simulator.beam.size_mm = 0.001
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

```
srun -n 40 -c 2 simtbx.diffBragg.prediction  stage_one_folder=stage_one oversample_override=1 Ncells_abc_override=[7,7,7] njobs=1 d_min=2  pandas_outfile=stage_one.pkl
```

The output pandas dataframe pickle is a suitable input for stage_two (global) diffBragg refinement. It contains the paths to the
optimized experiments and the corresponding diffBragg prediction reflection tables. 


## Stage 2: Structure factor refinement

Using the output file from the predictions script we can run structure factor refinement. In this case, we know the ground
truth structure factors, therefore we can use them as a reference to observe how accurate the optimization is with each 
iteration. The following python script can be used to generate the ground truth structure factor MTZ file. The PDB file
should be in the sim folder of this repository, however it can be downloaded from the PDB using the command 
```iotbx.fetch_pdb 4bs7```. Also, the file scanned_fp_fdp.tsv represents the anomlous contribution to the scattering, and is 
found in the ```sf``` folder of this repository: 

```python
from cxid9114.sf import struct_fact_special
from cxid9114.parameters import WAVELEN_HIGH
Famp = struct_fact_special.sfgen(WAVELEN_HIGH, "4bs7.pdb", dmin=1.9, yb_scatter_name="scanned_fp_fdp.tsv")
Famp = Famp.as_amplitude_array()
mtz = Famp.as_mtz_dataset(column_root_label="F").mtz_object()
mtz.write("cxid9114_grndtruth.mtz")
```

```
OMP_NUM_THREADS=2 time srun -n 40 -c 2 simtbx.diffBragg.stage_two stage_two.phil  io.output_dir=stage_two pandas_table=stage_one.pkl
```

where ```stage_two.phil``` contains the following

```
roi.shoebox_size = 15
roi.fit_tilt = True
roi.reject_edge_reflections = False
roi.pad_shoebox_for_background_estimation=0
refiner.refine_Fcell = [1]
refiner.refine_spot_scale = [1]
refiner.max_calls = [450]
refiner.sensitivity.spot_scale = 1
refiner.sensitivity.fcell = 1
refiner.ncells_mask = 111
refiner.tradeps = 1e-20
refiner.verbose = 0
refiner.sigma_r = 3
refiner.adu_per_photon = 28
refiner.stage_two.print_reso_bins = True
refiner.stage_two.merge_stat_freq = 1
refiner.stage_two.Fref_mtzname = cxid9114_grndtruth.mtz
refiner.stage_two.Fref_mtzcol = "F(+),F(-)"
simulator.crystal.has_isotropic_ncells = True
simulator.structure_factors.mtz_name = some_images_merged/iobs_all.mtz 
simulator.structure_factors.mtz_column = "Iobs(+),SIGIobs(+),Iobs(-),SIGIobs(-)"
simulator.beam.size_mm = 0.001
```

The output folder ```stage_two``` contains information necssary to construct the optimized miller array

```python
import numpy as np
from cctbx.array_family import flex
from cctbx import miller
from cctbx.crystal import symmetry

# load the asu mapping and the last iteration fcell file from the stage_two output folder:
asu_map = np.load("f_asu_map.npy", allow_pickle=True)[()]
Famp = np.load("_fcell_trial0_iter429.npz")["fvals"]

# construct a symmetry object for this lysozyme crystal:
sym = symmetry((79.1,79.1,38.4,90,90,90), "P43212")
val_at_index = {hkl: Famp[i] for i,hkl in asu_map.items()}
hkl_list = list(asu_map.values())
# construct a miller array object and save to mtz
millset = miller.set(sym, flex.miller_index(hkl_list), anomalous_flag=True)
mill_ary = millset.array(data=flex.double([val_at_index[hkl] for hkl in hkl_list]  ))
print(mill_ary.completeness())  # shows the data completeness
mtz = mill_ary.as_mtz_dataset("Famp").mtz_object()
mtz.write("optimized.mtz")
```

## Optional GPU support for diffBragg

This is still in early development, but GPU acceleration can help in cases where one includes a full spectrum in the 
modeling. Assuming you configured CCTBX with cuda support (as shown above), from a clean test folder you can

```
export DIFFBRAGG_USE_CUDA=1
libtbx.run_tests_parallel module=simtbx nproc=1
unset DIFFBRAGG_USE_CUDA # IMPORTANT!
```

You will notice a few tests fail (5 at the time of writing), and that is ok, as they require curvature analysis which is currently not 
supported using the GPU. GPU usage is controlled using an environment variable and some extra phil arguments. Note, to use GPU
refinement for stage_two , the above command would be

```
DIFFBRAGG_USE_CUDA=1 DIFFBRAGG_NUM_BLOCKS=128 time srun -n 24 -c 2 simtbx.diffBragg.stage_two stage_two.phil  io.output_dir=stage_two_gpu pandas_table=stage_one_again.pkl num_devices=8 randomize_devices=True
``` 

The above is a suitable command for a NERSC GPU node (pre-perlmutter), that has 8 GPUs per rank. The randomize_devices flag
will pick a GPU at random each time a rank runs diffBragg. In this particular example, GPU doesnt offer much in terms of speedup
because the model is monochromatic, however in certain cases where per-shot energy spectra are included in the model, GPU acceleration is
at least 100-500 fold depending on the problem. Low-level control over the GPU is provided in the API (if developing your own refinement
engine): 

```python
# D is a diffBragg instance (in python)
D.use_cuda = True  # will try to use GPU acceleration wherever possible
D.verbose = 1 # will print whether the GPU or CPU kernel was exectuted 
D.gpu_free()  # frees allocated GPU memory
```

