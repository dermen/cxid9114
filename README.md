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
* **~/Crystal/mc3** (conatains the conda install, assuming no other conda was in the path, and/or no miniconda install folders were found in the home directory, ```source ~/Crystal/mc3/etc/profile.d/conda.sh```)
* **~/Crystal/conda_base** (contains the conda environment, can be activated using ```conda activate ~/Crystal/conda_base```, but thats not necessary to use CCTBX or DIALS) 

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
```

### Adding some extra python modules

```bash
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
srun -N2 -n2 -c2 libtbx.python ~/test_mpi.py
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

To generate the two images (indicated by ```-trials 2```) it took 2min 40sec on a macbook. The images simulated for the paper included ```--mos_doms 50```, so expect that to take 100x longer to simulate (2x per mosaic block, as mosaic blocks are simulated in pairs to form an even distribution, such that the average mosaic missetting angle is 0 deg). We generated all images for the paper on a GPU node at the NERSC supercomputer. If you built CCTBX with cuda enabled, then you can also run the GPU simulation by adding the arguments ```-g 1  --gpu```, where ```-g``` specifies the number of GPU devices on the compute node. This will make simulating the images much faster. Also, the script can be run using MPI, parallelizing over images. The full command used at NERSC (1 compute node with 20 MPI ranks utilizing 8 GPUs) was 

```bash
srun -n 20 -c 2 libtbx.python d9114_mpi_sims.py  -o test -odir some_images --add-bg --add-noise --profile gauss --bg-name mybackground.h5 -trials 2000  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 50 --mos_spread_deg 0.01  --saveh5 --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115 -g 8 --gpu
```

MPI Rank 0 will monitor GPU usage by periodically printing the result of nvidia-smi to the screen.

The simulated images can be opened with the DIALS image viewer. Note, the spots are large - this is because we intensionally simulated small mosaic domains in order to reduce aliasing errors that arise for Bragg spots that are much smaller than the solid angle subtended by a pixel. In such a case, one needs to increase the oversample factor to ensure proper sampling within each pixel, however this adds to the overall computational time (increasing the oversample from e.g. 1 to 5 incurs a 125x [!] increase in run-time).  


# Process the images with DIALS

## Indexing and integration

We will index and integrate the images using ```dials.stills_process```. 

```
cd /path/to/some_images
dials.stills_process process.phil  job*/test_*.h5 mp.nproc=5 output.output_dir=~/Crystal/index
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

Indexing will output reflection files and experiment files. For a detailed discription of these files, see the references by Brewster linked to in the manuscript. Here, we use the parameter ```set_domain_size_ang_value``` in order to intentionally over-predict the diffraction pattern. These over-predictions will provide more information for the diffBragg refinement. 

## Merge the data

To merge the data, run the command

```
mkdir ~/Crystal/merge # make an output folder for merge results
srun  -n5 -c2 cctbx.xfel.merge merge.phil input.path=~/Crystal/index output.output_dir=~/Crystal/merge
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

The merge result is now in ```~/Crystal/merge/iobs_all.mtz```, and the log file containing merge statistics is ```~/Crystal/merge/iobs_main.log```.

# Process the images with diffBragg

The program diffBragg is currently part of a feature branch of CCTBX. It is actively under development, and ultimately it will be merged into the master branch of CCTBX. For now, if you wish to use diffBragg, you must checkout the feature branch (this instructional will be updated once diffBragg is moved into CCTBX master):

```
cd ~/Crystal/modules/cctbx_project
git checkout diffBragg
cd ~/Crystal/build
make

# go to an empty test folder and run the tests
cd ~/Crystal/tests
libtbx.run_tests_parallel module=simtbx nproc=5
```

If you configured CCTBX to use enable CUDA *as shown above* then you can test the GPU functionality. There is a single environment variable to turn this on, see below.

## Process with simtbx.diffBragg.stage_one

```
srun -n40 -c2 simtbx.diffBragg.stage_one  ~/Crystal/index/idx-*integrated.refl some_images_indexed/idx-*integrated.expt  stage_one.phil  save.pandas=True save.reflections=True save.experiments=True output.directory=some_images_indexed_optimized max_calls=[100,100] usempi=True 
```

where the files ```~/Crystal/index/idx-*integrated.refl``` are the integration files produced by dials.stills_process, and the phil file contains the following

```
roi.shoebox_size = 15
roi.fit_tilt = True
roi.reject_edge_reflections = False
roi.pad_shoebox_for_background_estimation=0
refiner.refine_Umatrix = [0,1]
refiner.refine_ncells = [1,0]
refiner.refine_spot_scale = [1,0]
refiner.refine_Bmatrix = [0,1]
refiner.max_calls = [1000, 1000]
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

Note, the file ```iobs_all.mtz``` is the merge output from running ```cctbx.xfel.merge```, as shown above. It is the initial guess of the structure factors.  

## Optional GPU support for diffBragg

This is still in early development, but GPU acceleration can help in cases where one includes a full spectrum in the modeling. Assuming you configured CCTBX with cuda support (as shown above), from a clean test folder you can

```
export DIFFBRAGG_USE_CUDA=1
libtbx.run_tests_parallel module=simtbx nproc=1
unset DIFFBRAGG_USE_CUDA # IMPORTANT!
```

You will notice a handfull of tests fail, and that is ok, as they require curvature analysis which is currently not supported using the GPU. Note, the flag ```DIFFBRAGG_USE_CUDA``` is not fully supported, and there is better ways to control GPU usage, for example

```
# D is a diffBragg instance (in python)
D.use_cuda = True  # will try to use GPU acceleration wherever possible
D.gpu_free()  # frees allocated GPU memory
```

Also, the script ```simtbx.diffBragg.stage_one``` accepts a ```use_cuda=True``` flag