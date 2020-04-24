
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("-N", type=int, default=1000, help="Number of spectra to generate")
parser.add_argument("--flux", type=float, default=8e10, help="average spectra flux")
parser.add_argument("-o", type=str, required=True, help="output file name")
args = parser.parse_args()

# coding: utf-8
from LS49.spectra.generate_spectra import spectra_simulation
import h5py
import numpy as np
from cxid9114.parameters import ENERGY_HIGH

SS = spectra_simulation()
total_flux = args.flux
Nspec = args.N
E = ENERGY_HIGH
#E = 9476  # cyto data
#SS.generate_recast_renormalized_images)2-, energy=
spectras = SS.generate_recast_renormalized_images(
        Nspec, energy=E, total_flux=total_flux)
outs = []
for out in spectras:
    outs.append(out)
    
waves, fluxes, ebeams = zip(*outs)
waves = np.vstack(waves)
fluxes = np.vstack(fluxes)
ebeams = np.array(ebeams)
comp_args = {"compression":"gzip", 
    "compression_opts":9, "shuffle":True}

from cxid9114.parameters import ENERGY_CONV
energies = np.vstack([ENERGY_CONV / W for W in waves])
with h5py.File(args.o, "w") as f:
    f.create_dataset("wavelengths", data=waves.astype(np.float32), **comp_args)
    f.create_dataset("energies", data=energies.astype(np.float32), **comp_args)
    f.create_dataset("fluxes", data=fluxes.astype(np.float32), **comp_args)
    f.create_dataset("wave_ebeams", data=ebeams.astype(np.float32), **comp_args)

