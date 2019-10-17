
from argparse import ArgumentParser
from copy import deepcopy

parser = ArgumentParser("Make prediction boxes")

parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--nrank", type=int, default=1)
parser.add_argument("--glob", type=str, required=True, help="experiment list glob")
parser.add_argument("-o",help='output directoty',  type=str, default='.')
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
from simtbx.diffBragg.utils import tilting_plane


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

GAIN = 28

if rank==0:
    print El_fnames
all_paths = []
all_Amats = []
odir = args.o

writer = h5py.File(os.path.join( odir, "process_rank%d.h5" % rank), "w")

n_processed = 0
for i_shot,(El_json, refl_pkl) in enumerate(zip(El_fnames, refl_fnames)):
    if i_shot % size != rank:
        continue
    if rank==0:
        print("Rank 0: Doing shot %d / %d" % (i_shot+1, len( El_fnames)))

    El = ExperimentListFactory.from_json_file(El_json, check_format=False)
    
    iset = El.imagesets()[0]
    fpath = iset.get_path(0)
    h5 = h5py.File( fpath.replace(".npz", ""), 'r')
    mos_spread = h5["mos_spread"][()]
    Ncells_abc = tuple(h5["Ncells_abc"][()])
    mos_doms = h5["mos_doms"][()]
    profile = h5["profile"][()]
    beamsize = h5["beamsize_mm"][()]
    exposure_s = h5["exposure_s"][()]
    spectrum = h5["spectrum"][()]
    total_flux = np.sum(spectrum) 
    xtal_size = 0.0005 #h5["xtal_size_mm"][()]

    refls_data = utils.open_flex(refl_pkl)

    # Make a strong spot mask that is used to fit tilting planes
    img = np.load(fpath)["img"]
    img_shape = img.shape
    is_bg_pixel = np.ones(img_shape, bool)
    # mask the strong spots so they dont go into tilt plane fits
    for x1,x2,y1,y2,_,_ in refls_data['bbox']:
        bb_ss = slice(y1,y2,1)
        bb_fs = slice(x1,x2,1)
        is_bg_pixel[bb_ss, bb_fs] = False

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

    # this gets the integration shoeboxes, not to be confused with strong spot bound boxes
    Hi, bboxes = prediction_utils.get_prediction_boxes(refls_at_colors, 
                DET, beams, crystal, delta_q=0.0475) # ret_patches=True, fc='none', ec='w')

    # fit the background plane
    tilt_abc = []
    for i1,i2,j1,j2 in bboxes:
        shoebox_img = img[j1:j2, i1:i2] / GAIN  # NOTE: gain is imortant here!
        # TODO: Consider moving the bg fitting to the instantiation of the Refiner..
        # TODO: ...to protect the case when bg planes are fit to images without gain correction
        shoebox_mask = is_bg_pixel[j1:j2, i1:i2]
        tilt, bgmask, coeff, _ = tilting_plane(
            shoebox_img,
            mask=shoebox_mask,  # mask specifies which spots are bg pixels...
            zscore=2)  # zscore is standard M.A.D. score for removing additional zingers from the fit
        # note this fit should be EXACT, linear regression..
        tilt_abc.append((coeff[1], coeff[2], coeff[0]))  # store as fast-scan coeff, slow-scan coeff, offset coeff

    all_paths.append(fpath)
    all_Amats.append(crystal.get_A())
    if rank==0:
        print("Rank0: writing")
    writer.create_dataset("bboxes/shot%d" % n_processed, data=bboxes,  dtype=np.int, compression="lzf" )
    writer.create_dataset("tilt_abc/shot%d" % n_processed, data=tilt_abc,  dtype=np.float32, compression="lzf" )
    writer.create_dataset("Hi/shot%d" % n_processed, data=Hi, dtype=np.int, compression="lzf")
    #writer.create_dataset("bg_pixel_mask/shot%d" % n_processed, data=is_bg_pixel, dtype=bool, compression="lzf")
    if rank==0:
        print("Rank0: Done writing")
    n_processed += 1

writer.create_dataset("Amatrices", data=all_Amats, compression="lzf")
writer.create_dataset("h5_path", data=all_paths, compression="lzf")
writer.close()

