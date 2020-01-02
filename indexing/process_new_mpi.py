#!/usr/bin/env libtbx.python

GAIN = 28
sigma_readout = 3

from argparse import ArgumentParser

parser = ArgumentParser("Make prediction boxes")

parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--pearl", action="store_true")
parser.add_argument("--pause", type=float, default=0.5)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--savefigdir", default=None, type=str)
parser.add_argument("--glob", type=str, required=True, help="experiment list glob")
parser.add_argument("--Z", type=float, default=2)
parser.add_argument("--deltaq", type=float, default=0.07)
parser.add_argument("--dilate", default=1, type=int)
parser.add_argument("--bgname", type=str, default=None)
parser.add_argument("--defaultF", type=float, default=1e3)
parser.add_argument("--sbpad", type=int, default=0, help="padding for background tilt plant fit")
parser.add_argument("--spline", action="store_true")
parser.add_argument("--sanitycheck", action="store_true")
parser.add_argument("--thresh", type=float, default=1e-2)
parser.add_argument("-o", help='output directoty',  type=str, default='.')
parser.add_argument("--miller", type=str, default=None, choices=["bs7", "p9", "datasf"])
parser.add_argument("--plot", default=None, type=float )
parser.add_argument("--plottilt", default=None, type=float )
parser.add_argument("--usegt", action="store_true")
parser.add_argument("--usepredictions", action="store_true")
parser.add_argument("--show_params", action='store_true')
parser.add_argument("--forcelambda", type=float, default=None)
parser.add_argument("--noiseless", action="store_true")
parser.add_argument("--symbol", default="P43212", type=str)
parser.add_argument("--sanityplots", action='store_true')
args = parser.parse_args()

import glob
import os
from copy import deepcopy

import numpy as np
import h5py
from scipy.ndimage.morphology import binary_dilation
from IPython import embed

from dxtbx.model.experiment_list import ExperimentListFactory, Experiment
from cctbx import miller, sgtbx
#from scitbx.array_family import flex
from dials.array_family import flex
from cctbx.crystal import symmetry
from cxid9114.sim import sim_utils
from cxid9114.geom.multi_panel import CSPAD
from cxid9114 import parameters, utils
from cxid9114.prediction import prediction_utils
from cxid9114.sf import struct_fact_special
from cxid9114.parameters import WAVELEN_HIGH
from tilt_fit.tilt_fit import tilt_fit

n_gpu = args.ngpu

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.size
    has_mpi = True
    rank = comm.rank
except ImportError:
    size = 1
    rank = 0
    has_mpi = False

if rank == 0:
    print(args)
    if args.sanityplots:
        import pylab as plt

# Load in the reflection tables and experiment lists
Els = glob.glob(args.glob)
El_fnames, refl_fnames, refl_indexed_fnames = [], [], []
for El_f in Els:
    name_base = El_f.split("_refined.expt")[0]

    refl_f = "%s_strong.refl" % name_base
    if not args.usepredictions:
        refl_idx_f = "%s_indexed.refl" % name_base
        if not os.path.exists(refl_idx_f):
            continue
    if os.path.exists(refl_f):
        El_fnames.append(El_f)
        refl_fnames.append(refl_f)
        if not args.usepredictions:
            refl_indexed_fnames.append(refl_idx_f)

if args.noiseless:
    GAIN = 1

if not os.path.exists(args.o) and rank == 0:
    os.makedirs(args.o)

MPI.COMM_WORLD.Barrier()

# load the bs7 default array
bs7_mil_ar = struct_fact_special.sfgen(WAVELEN_HIGH, "../sim/4bs7.pdb", yb_scatter_name="../sf/scanned_fp_fdp.npz")
datasf_mil_ar = struct_fact_special.load_4bs7_sf()

assert El_fnames
if rank == 0:
    print("I found %d fname" % len(El_fnames))
all_paths = []
all_Amats = []
odir = args.o

if args.bgname is not None:
    background = h5py.File(args.bgname, "r")['bigsim_d9114'][()]

writer = h5py.File(os.path.join(odir, "process_rank%d.h5" % rank), "w")

n_processed = 0
for i_shot, (El_json, refl_pkl) in enumerate(zip(El_fnames, refl_fnames)):
    if i_shot % size != rank:
        continue
    if rank == 0:
        print("Rank 0: Doing shot %d / %d" % (i_shot + 1, len(El_fnames)))

    # get the experiment stuffs
    El = ExperimentListFactory.from_json_file(El_json, check_format=True)
    # El = ExperimentListFactory.from_json_file(El_json, check_format=False)
    iset = El.imagesets()[0]
    fpath = iset.get_path(0)
    # this is the file containing relevant simulation parameters..
    h5 = h5py.File(fpath.replace(".npz", ""), 'r')
    # get image pixels
    _fpath = fpath
    noiseless_fpath = _fpath.replace(".npz", ".noiseless.npz")

    if args.debug:
        assert os.path.exists(noiseless_fpath)
        imgs_noiseless = np.load(noiseless_fpath)["img"]
    if args.noiseless:
        # load the image that has no noise
        assert os.path.exists(noiseless_fpath)
        img_data = np.load(noiseless_fpath)["img"]
    else:
        img_data = np.load(_fpath)["img"]
    # get simulation parameters
    mos_spread = h5["mos_spread"][()]
    Ncells_abc = tuple(h5["Ncells_abc"][()])
    mos_doms = h5["mos_doms"][()]
    profile = h5["profile"][()]
    beamsize = h5["beamsize_mm"][()]
    exposure_s = h5["exposure_s"][()]
    spectrum = h5["spectrum"][()]
    total_flux = np.sum(spectrum)
    xtal_size = 0.0005  #h5["xtal_size_mm"][()]

    # <><><><><><><><><><><><><><
    # HERE WE WILL DO PREDICTIONS
    # <><><><><><><><><><><><><><
    # make a sad spectrum
    FLUX = [total_flux]
    # loading the beam  (this might have wrong energy)
    BEAM = El.beams()[0]
    if args.forcelambda is None:
        ave_wave = BEAM.get_wavelength()
    else:
        ave_wave = args.forcelambda
    energies = [parameters.ENERGY_CONV/ave_wave]

    # grab the detector
    DET = El.detectors()[0]
    detdist = abs(DET[0].get_origin()[-1])
    pixsize = DET[0].get_pixel_size()[0]
    fs_dim, ss_dim = DET[0].get_image_size()
    n_panels = len(DET)
    # grab the crystal
    crystal = El.crystals()[0]

    # Optionally use the ground truth crystal A matrix and detector for prediction (only possible with simulations)
    if args.usegt:
        crystal.set_A(h5["crystalA"][()])
        DET = deepcopy(CSPAD)   # and use the ground truth CSPAD

    # make the miller array to be used with prediction
    sgi = sgtbx.space_group_info(args.symbol)
    # TODO: allow override of ucell
    symm = symmetry(unit_cell=crystal.get_unit_cell(), space_group_info=sgi)
    miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
    # NOTE does build_miller_set automatically expand to p1 ? Does it obey systematic absences ?
    # Note how to handle sys absences here ?
    Famp = flex.double(np.ones(len(miller_set.indices())) * args.defaultF)
    mil_ar = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()
    FF = [mil_ar]
    # Optionally use a miller array from a model to do predictions, as opposed to a flat miller array
    if args.miller is not None:
        if args.miller == "bs7":
            FF = [bs7_mil_ar]
        if args.miller == "datasf":
            FF = [datasf_mil_ar]

    #   <><><><><><><><><><><><><><><><><><><><>
    #   DO THE SIMULATION TO USE FOR PREDICTIONS
    #   <><><><><><><><><><><><><><><><><><><><>
    # choose a device Id for GPU
    device_Id = np.random.choice(range(n_gpu))
    # call the simulation helper
    simsAB = sim_utils.sim_colors(
        crystal, DET, BEAM, FF,
        energies,
        FLUX, pids=None, profile=profile, cuda=True, oversample=1,
        Ncells_abc=Ncells_abc, mos_dom=50, mos_spread=0.02,
        master_scale=1,
        exposure_s=exposure_s, beamsize_mm=beamsize, device_Id=device_Id,
        show_params=args.show_params, accumulate=False, crystal_size_mm=xtal_size)

    assert len(energies) == 1  # sanity check TODO remove

    # make a reflection table from the simulations, using a simple threshold
    refls_predict = prediction_utils.refls_from_sims(simsAB[0], DET, BEAM, thresh=args.thresh)
    if not refls_predict:
        print("refls_predictANK %d no spots!" % rank)
        continue
    n_predict = len(refls_predict)

    # make the spot integration foreground mask from the predictions
    panel_integration_masks = {}
    for pid in range(n_panels):
        panel_integration_masks[pid] = None
    # group predictions bty panel name
    refls_predict_bypanel = prediction_utils.refls_by_panelname(refls_predict)
    for panel_id in refls_predict_bypanel:
        fast, slow = DET[panel_id].get_image_size()
        mask = prediction_utils.strong_spot_mask_dials(refls_predict_bypanel[panel_id], (slow, fast),
                                      as_composite=True)
        # if the panel mask is not set, set it!
        if panel_integration_masks[panel_id] is None:
            panel_integration_masks[panel_id] = mask
        # otherwise add to it
        else:
            panel_integration_masks[panel_id] = np.logical_or(mask, panel_integration_masks[panel_id])

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # HERE WE LOAD THE STRONG SPOTS AND MAKE THEM INTO A MASK
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # load strong spot reflections
    refls_strong = flex.reflection_table.from_file( refl_pkl) #utils.open_flex(refl_pkl)
    # make mask of all strong spot pixels..
    nfast, nslow = DET[0].get_image_size()
    img_shape = nslow, nfast  # numpy format
    # make a mask that tells me True if I am a background pixel
    is_bg_pixel = np.ones((n_panels, nslow, nfast), bool)
    # group the refls by panel ID
    refls_strong_perpan = prediction_utils.refls_by_panelname(refls_strong)
    for panel_id in refls_strong_perpan:
        fast, slow = DET[panel_id].get_image_size()
        mask = prediction_utils.strong_spot_mask_dials(
            refls_strong_perpan[panel_id], (slow, fast),
            as_composite=True)
        # dilate the mask
        mask = binary_dilation(mask, iterations=args.dilate)
        is_bg_pixel[panel_id] = ~mask  # strong spots should not be background pixels


    # Combine strong spot mask and integration mask, both with there dilations, to get the best
    # possible selection of background pixels..
    for i_predict in range(n_predict):
        ref_predict = refls_predict[i_predict]
        i1, i2, j1, j2, _, _ = ref_predict['bbox']
        i_panel = ref_predict['panel']
        integration_mask = panel_integration_masks[i_panel][j1:j2, i1:i2]
        # expand the integration mask so as not to include background pixels near it
        expanded_integration_mask = binary_dilation(integration_mask, iterations=args.dilate)

        # get the pixels already marked as background
        bg = is_bg_pixel[i_panel, j1:j2, i1:i2]
        # update the background pixel selection with the expanded integration mask
        is_bg_pixel[i_panel, j1:j2, i1:i2] = ~np.logical_or(~bg, expanded_integration_mask)
    # At this point is_bg_pixel returns False for pixels that are inside the expanded strong spot mask
    # or the expanded integration mask

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    #   RUN THE TILT PLANE HELPER FUNCTION
    # 1. Weighted fit of the background tilt plane
    # 2. Updates prediction reflection table with miller indices, and shoeboxes
    # 3. Updates prediction reflections with integrations and integration variances
    #    using Leslie 99
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    exper = Experiment()
    exper.detector = DET
    exper.beam = BEAM
    exper.crystal = crystal
    exper.imageset = El[0].imageset
    results = tilt_fit(
        imgs=img_data, is_bg_pix=is_bg_pixel,
        delta_q=args.deltaq, photon_gain=GAIN, sigma_rdout=sigma_readout, zinger_zscore=args.Z,
        exper=exper, predicted_refls=refls_predict, sb_pad=args.sbpad)

    refls_predict, tilt_abc, error_in_tilt, I_Leslie99, varI_Leslie99 = results
    spot_snr = np.array(I_Leslie99) / np.sqrt(varI_Leslie99)
    shoeboxes = refls_predict['shoebox']
    bboxes = np.vstack([list(sb.bbox)[:4] for sb in shoeboxes])
    bbox_panel_ids = np.array(refls_predict['panel'])
    Hi = np.vstack(refls_predict['miller_index'])
    did_i_index = np.array(refls_predict['id'])
    boundary_spot = np.array(refls_predict['boundary'])

    # <><><><><><><><><><><><><><><>
    # DO THE SANITY PLOTS (OPTIONAL)
    # <><><><><><><><><><><><><><><>
    if rank == 0 and args.sanityplots:
        refls_predict_bypanel = prediction_utils.refls_by_panelname(refls_predict)
        plt.figure()
        pause = args.pause
        for panel_id in refls_predict_bypanel:
            panel_img = img_data[panel_id]
            m = panel_img.mean()
            s = panel_img.std()
            vmax = m + 4*s
            vmin = m - s
            plt.cla()
            im = plt.imshow(panel_img, vmax=vmax, vmin=vmin)
            int_mask = np.zeros(panel_img.shape).astype(np.bool)
            bg_mask = np.zeros(panel_img.shape).astype(np.bool)

            for i_ref in range(len(refls_predict_bypanel[panel_id])):
                ref = refls_predict_bypanel[panel_id][i_ref]
                i1, i2, j1, j2, _, _ = ref['shoebox'].bbox
                rect = plt.Rectangle(xy=(i1, j1), width=i2-i1, height=j2-j1, fc='none', ec='Deeppink')
                plt.gca().add_patch(rect)
                mask = ref['shoebox'].mask.as_numpy_array()[0]
                int_mask[j1:j2, i1:i2] = np.logical_or(mask == 5, int_mask[j1:j2, i1:i2])
                bg_mask[j1:j2, i1:i2] = np.logical_or(mask == 19, bg_mask[j1:j2, i1:i2])
            plt.draw()
            plt.pause(pause)
            im.set_data(int_mask)
            plt.title("panel%d: integration mask" % i_panel)
            im.set_clim(0, 1)
            plt.draw()
            plt.pause(pause)
            im.set_data(bg_mask)
            plt.title("panel%d: background mask" % i_panel)
            im.set_clim(0, 1)
            plt.draw()
            plt.pause(pause)

    all_paths.append(fpath)
    all_Amats.append(crystal.get_A())
    if rank == 0:
        print("Rank0: writing")
    # save the output!
    writer.create_dataset("bboxes/shot%d" % n_processed, data=bboxes,  dtype=np.int, compression="lzf" )
    writer.create_dataset("tilt_abc/shot%d" % n_processed, data=tilt_abc,  dtype=np.float32, compression="lzf" )
    writer.create_dataset("tilt_error/shot%d" % n_processed, data=error_in_tilt,  dtype=np.float32, compression="lzf" )
    writer.create_dataset("SNR_Leslie99/shot%d" % n_processed, data=spot_snr, dtype=np.float32, compression="lzf" )
    writer.create_dataset("Hi/shot%d" % n_processed, data=Hi, dtype=np.int, compression="lzf")
    writer.create_dataset("indexed_flag/shot%d" % n_processed, data=did_i_index, dtype=np.int, compression="lzf")
    writer.create_dataset("is_on_boundary/shot%d" % n_processed, data=boundary_spot, dtype=np.bool, compression="lzf")
    writer.create_dataset("panel_ids/shot%d" % n_processed, data=bbox_panel_ids, dtype=np.int, compression="lzf")
    # add the default peak selection flags (default is all True, so select all peaks for refinement)
    keepers = np.ones(len(bboxes)).astype(np.bool)
    writer.create_dataset("bboxes/keepers%d" % n_processed, data=keepers, dtype=np.bool, compression="lzf")

    if rank == 0:
        print("Rank0: Done writing")
    n_processed += 1

writer.create_dataset("Amatrices", data=all_Amats, compression="lzf")
writer.create_dataset("h5_path", data=all_paths, compression="lzf")
writer.close()

