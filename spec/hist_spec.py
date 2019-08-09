
import numpy as np
import h5py
import sys

fname = "realspec.h5"
raw_spec_path = "spectra"
f = h5py.File(fname, "r+")

# normalize with this flux level
ave_fluence = 2e11
# resolution of output spectrum
ev_width = 1
# this scales the FEE x-axis to electron volts
Efit = np.array([1.45585875e-01, 8.92420032e+03])

# new h5 keys for histogrammed spectra and energy channel eV values
proc_spec_name = "hist_spec"
energy_name = "energy_bins"
# remove them if they exist
if proc_spec_name in f:
    del f[proc_spec_name]
if energy_name in f:
    del f[energy_name]

f.close()
exit()

# load the raw spec convert to float32 from float16
raw_specs = f[raw_spec_path][()].astype(np.float32)
Nspec_bins = raw_specs.shape[1]  # 1024
assert( Nspec_bins == 1024)
Nspecs = raw_specs.shape[0]  # number of spectra

# the scaled x-axis in its raw resolution
Edata = np.polyval( Efit, np.arange( Nspec_bins))
en_bins = np.arange( Edata[0],Edata[-1]+1, ev_width)
en_bin_cent = 0.5*en_bins[1:] + 0.5*en_bins[:-1]
spec_hists = np.zeros((Nspecs, en_bin_cent.shape[0]))
for i_spec, spec in enumerate(raw_specs):
    if i_spec %50==0:
        print ('processing spec %d / %d') % (i_spec+1, Nspecs)
    spec_hists[i_spec] = np.histogram(Edata, en_bins, weights=spec)[0]

# normalize the spectra to the desired average fluence
K = ave_fluence/(spec_hists.sum(axis=1).mean())
f.create_dataset(proc_spec_name, data=spec_hists*K, compression="lzf")
f.create_dataset(energy_name, data=en_bin_cent, compression="lzf")
f.close()

