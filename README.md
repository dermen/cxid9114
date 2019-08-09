## D9114 instructions on CORI

This is tested on SUSE linux:

```
$ lsb_release  -a
LSB Version:	n/a
Distributor ID:	openSUSE
Description:	openSUSE Leap 15.0
Release:	15.0
Codename:	n/a
```

### STEP 1
Download/install a copy of go language


```
mkdir ~/install_go
cd ~/install_go
wget https://dl.google.com/go/go1.12.7.linux-amd64.tar.gz
tar -xzvf go1.12.7.linux-amd64.tar.gz
export PATH=$PATH:~/install_go/go/bin
```
### STEP 2
Install git lfs (large file storage)

```
cd # go to a home directory
go get github.com/git-lfs/git-lfs
export PATH=$PATH:~/go/bin
```

### STEP 3
Clone cxid9114 and bring in big files

```
cd # home directory
git clone https://github.com/dermen/cxid9114.git
cd ~/cxid9114
git lfs install --local
git lfs fetch
git lfs pull
```

### STEP 4
Build cctbx with nvcc enabled in a working directory, e.g. ```~/crystal``` 

```
source ~/crystal/build/setpaths.sh
```

Then link the cxid9114 repo to cctbx and configure it

```
ln -s ~/cxid9114 ~/crystal/modules
libtbx.configure cxid9114
```

### STEP 5
Histogram the spectra from there raw format to a 1eV spacing:

```
cd ~/cxid9114/spec
libtbx.python hist_spec.h5
```

Compute structure factors for each energy channel in the histogrammed spectra:

```
cd ~/cxid9114/sf
libtbx.python struct_fact_special.py
```

### Step 6
Install format class for reading simulated images

```
cd ~/cxid9114/format
dxtbx.install_format -u FormatBigSimD9114.py
```

### Step 7
Access a CORI GPU node

```
# Log into CORI
$ ssh -i ~/.ssh/nersc -Y rand_al_thor@cori.nersc.gov
$ salloc  -C gpu -c 10 -t 10 -A m1759 --gres=gpu:1 -N 1
salloc: Granted job allocation 196284
salloc: Waiting for resource configuration
salloc: Nodes cgpu01 are ready for job
```

### Notes
If you want to push to the cxid9114 repo, you will always have to put ```git-lfs``` in your path (see e.g. step 2 above)
