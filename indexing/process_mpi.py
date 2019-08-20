
from argparse import ArgumentParser
from copy import deepcopy

parser = ArgumentParser("Make prediction boxes")

parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--nrank", type=int, default=1)
parser.add_argument("--glob", type=str, required=True)
parser.add_argument("--show_params", action='store_true')
args = parser.parse_args()

import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
from mpi4py import MPI
import h5py

from cxid9114.sim import sim_utils
from cxid9114.geom.single_panel import DET,BEAM
from cxid9114 import parameters, utils
from cxid9114.prediction import prediction_utils

import glob
import os

n_gpu = args.ngpu
size =  MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank

# Load in the reflection tables and experiment lists
El_fnames , refl_fnames = [],[]
for El_f in glob.glob(args.glob):
    refl_f = El_f.replace("El", "refl").replace(".json", ".pkl")
    if os.path.exists(refl_f):
        El_fnames.append( El_f)
        refl_fnames.append( refl_f)

if rank==0:
    print El_fnames
all_paths = []

writer = h5py.File("process_rank%d.h5" % rank, "w")

n_processed = 0
for i_shot,(El_json, refl_pkl) in enumerate(zip(El_fnames, refl_fnames)):
    if i_shot % size != rank:
        continue
    if rank==0:
        print("Rank 0: Doing shot %d / %d" % (i_shot+1, len( El_fnames)))

    El = ExperimentListFactory.from_json_file(El_json, check_format=False)

    iset = El.imagesets()[0]
    fpath = iset.get_path(0)
    h5 = h5py.File( fpath, 'r')
    mos_spread = 0.01#h5["mos_spread"][()]
    Ncells_abc = (20,20,20) #h5["Ncells_abc"][()]
    mos_doms = 120 #h5["mos_doms"][()]
    profile = "gauss" #h5["profile"][()]
    beamsize = h5["beamsize_mm"][()]
    exposure_s = h5["exposure_s"][()]
    spectrum = h5["spectrum"][()]
    total_flux = np.sum(spectrum) 
    xtal_size = 0.0005 # h5["xtal_size_mm"][()]

    refls_data = utils.open_flex(refl_pkl)
    FF = [1e3, None] # NOTE: not sure what to do here, we dont know the structure factor
    FLUX = [total_flux * .5, total_flux*.5]
    energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]

    crystal = El.crystals()[0]
    beams= []
    device_Id = rank % n_gpu 
    simsAB = sim_utils.sim_colors(
        crystal, DET, BEAM, FF,
        energies, 
        FLUX, pids=None, profile=profile, cuda=True, oversample=1,
        Ncells_abc=Ncells_abc, mos_dom=mos_doms, mos_spread=mos_spread,
        exposure_s=exposure_s, beamsize_mm=beamsize, device_Id=device_Id, 
        show_params=args.show_params, accumulate=False, crystal_size_mm=xtal_size)

    refls_at_colors = []
    for i_en, en in enumerate(energies):
        beam = deepcopy(BEAM)
        beam.set_wavelength( parameters.ENERGY_CONV/en)
        try:
            R=prediction_utils.refls_from_sims(simsAB[i_en], DET, BEAM, thresh=1e-2)
        except:
            continue
        refls_at_colors.append(R)
        beams.append(beam)

    if not refls_at_colors:
        continue

    Hi, bboxes = prediction_utils.get_prediction_boxes(refls_at_colors, 
                DET, beams, crystal, delta_q=0.0475) # ret_patches=True, fc='none', ec='w')

    all_paths.append(fpath)
    if rank==0:
        print("Rank0: writing")
    writer.create_dataset("bboxes/shot%d" % n_processed, data=bboxes,  dtype=np.int, compression="lzf" )
    writer.create_dataset("Hi/shot%d" % n_processed, data=Hi, dtype=np.int, compression="lzf")
    if rank==0:
        print("Rank0: Done writing")
    n_processed += 1

writer.create_dataset("h5_path", data=all_paths, compression="lzf")
writer.close()

