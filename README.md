# Manuscript work

For the work presented in [Beyond integration: modeling every pixel to obtain better structure factors from stills, IUCrJ Nov. 2020](https://doi.org/10.1107/S2052252520013007)


## Install CCTBX (optionally with CUDA support)

The below is for the NERSC GPU nodes, but it could easily be adapted to fit your local environment. 

##### Build sources

Grab the bootstrap script and execute bootstrap with any modern python interpreter

```bash
# For GPU builds
module load cgpu gcc openmpi cuda # on NERSC only
# Verify nvcc is in your path
nvcc --version

mkdir ~/Crystal
cd ~/Crsytal
wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
# if you arent doing a GPU build, then remove the config-flags argument
python bootstrap.py --builder=dials --use-conda --nproc=4 --config-flags="--enable_cuda" --python=38
```

This should create some sub-folders modules (contains the sources) build (contains the setup script). 

##### Test the build

```bash
# this sets up your environment to use CCTBX  and DIALS
source ~/Crystal/setpaths.sh
```

You can test the installation

```bash
mkdir ~/Crystal/test
cd ~/Crystal/test
libtbx.run_tests_parallel nproc=4 module=simtbx
```

##### Install the repository for the manuscript work

Now grab the cxid9114 repo

```bash
cd ~/Crsytal/modules # its important to place it in the modules folder
git clone https://github.com/dermen/cxid9114.git
# install git-lfs (if on nersc, just load the module
module load git-lfs
cd ~/Crystal/modules/cxid9114
git lfs install
git lfs fetch
git pull # this should bring some extra file content needed for the simulations
```

##### Adding some extra python modules

```bash
litbx.python -m pip install pandas jupyter
libtbx.refresh
libtbx.ipython # launch an interactive python shell
```

##### Install the image format

Multi-panel images simulated with nanoBragg are saved in a custom-written format (```simtbx.nanoBragg.utils.H5AttributeGeomWriter```). The format is simple: the images are stored as 3D hdf5 datasets, and the dxtbx detector and beam models are converted to json strings and stored in the hdf5 dataset attribute field. The format reader can be installed as follows:

```bash
cd ~/Crystal/modules/cxid9114/format
dxtbx.install_format  -u FormatHDF5AttributeGeometry.py
```

##### Install and test mpi4py

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

##### Make a background image

```bash
libtbx.python d9114_mpi_sims.py  -odir . --bg-name mybackground.h5 --make-background   --sad 
```

You can view the background after installing the necessary dxtbx format class (as shown above)

```bash
dials.image_viewer mybackground.h5
```

##### Make the diffracton patterns

Below is a script that can run on a PC to generate 2 diffraction images:

```bash
libtbx.python d9114_mpi_sims.py  -o test -odir some_images --add-bg --add-noise --profile gauss --bg-name mybackground.h5 -trials 2  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 1 --mos_spread_deg 0.01  --saveh5 --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115
```

To generate the two images (indicated by ```-trials 2```) it took 2min 40sec on a macbook. The images simulated for the paper included ```--mos_doms 50```, so expect that to take 100x longer to simulate (2x per mosaic block, as mosaic blocks are simulated in pairs to form an even distribution, such that the average mosaic missetting angle is 0 deg). We generated all images for the paper on a GPU node at the NERSC supercomputer. If you built CCTBX with cuda enabled, then you can also run the GPU simulation by adding the arguments ```-g 1  --gpu```, where ```-g``` specifies the number of GPU devices on the compute node. This will make simulating the images much faster. Also, the script can be run using MPI, parallelizing over images. The full command used at NERSC (1 compute node with 20 MPI ranks utilizing 8 GPUs) was 

```bash
srun -n 20 -c 2 libtbx.python d9114_mpi_sims.py  -o test -odir some_images --add-bg --add-noise --profile gauss --bg-name mybackground.h5 -trials 2000  --oversample 0 --Ncells 10 --xtal_size_mm 0.00015 --mos_doms 50 --mos_spread_deg 0.01  --saveh5 --readout  --masterscale 1150 --sad --bs7real --masterscalejitter 115 -g 8 --gpu
```

MPI Rank 0 will monitor GPU usage by periodically printing the result of nvidia-smi to the screen.

