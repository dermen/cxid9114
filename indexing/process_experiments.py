#!/usr/bin/env libtbx.python

from argparse import ArgumentParser

parser = ArgumentParser("Make prediction boxes")

parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--pearl", action="store_true")
parser.add_argument("--nocuda", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--showcompleteness", action="store_true")
parser.add_argument("--savefigdir", default=None, type=str)
parser.add_argument("--filteredexpt", type=str, required=True, help="filtered combined experiment file")
parser.add_argument("--filteredrefl", type=str, default=None, help="filtered combined reflections file")
parser.add_argument("--Z", type=float, default=2)
parser.add_argument("--deltaq", type=float, default=0.07)
parser.add_argument("--dilate", default=1, type=int)
parser.add_argument("--bgname", type=str, default=None)
parser.add_argument("--defaultF", type=float, default=1e3)
parser.add_argument("--sbpad", type=int, default=0, help="padding for background tilt plant fit")
parser.add_argument("--spline", action="store_true")
parser.add_argument("--sanitycheck", action="store_true")
parser.add_argument("--thresh", type=float, default=1e-2)
parser.add_argument("-o", help='output directoty', type=str, default='.')
parser.add_argument("--miller", type=str, default=None, choices=["bs7", "p9", "datasf"])
parser.add_argument("--usegt", action="store_true")
parser.add_argument("--show_params", action='store_true')
parser.add_argument("--forcelambda", type=float, default=None)
parser.add_argument("--noiseless", action="store_true")
parser.add_argument("--imgdirname", type=str, default=None)
parser.add_argument("--indexdirname", type=str, default=None)
parser.add_argument("--symbol", default="P43212", type=str)
parser.add_argument("--sanityplots", action='store_true')
parser.add_argument("--pause", type=float, default=0.5)
parser.add_argument("--bgestimate", type=str, required=True, help="path to estimated background")
args = parser.parse_args()

GAIN = 28
sigma_readout = 3
nmismatch = 0
nstrong_tot = 0
nstills_pred = 0
nbragg_pred = 0
bragg_hi = []
stills_hi = []

import glob
import os
from copy import deepcopy

import numpy as np
import h5py
from scipy.ndimage.morphology import binary_dilation
from scipy.optimize import minimize
from IPython import embed
from scipy.spatial import cKDTree


from dxtbx.model.experiment_list import ExperimentListFactory, Experiment
import time
from cctbx import miller, sgtbx
# from scitbx.array_family import flex
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

        fig = plt.figure()
        ax = plt.gca()

from cxid9114.geom.noHierarchy import CSPAD


def min_panel(coef, data, bg_est):
    residual = data - coef[0] * bg_est
    return np.sum(residual ** 2)

import dxtbx
BG_ESTIMATE = np.array([ I.as_numpy_array() for I in  dxtbx.load(args.bgestimate).get_raw_data(0)])


# Load in the reflection tables and experiment lists
print("Reading in the files")
El = ExperimentListFactory.from_json_file(args.filteredexpt, check_format=False)
refl_name = args.filteredrefl
if refl_name is None:
    refl_name = args.filteredexpt.replace(".expt", ".refl")
Rmaster = flex.reflection_table.from_file(refl_name)
Nexper = len(El)
print("Read int %d experiments" % Nexper)
# get the original indexing directory name
indexdirname = args.indexdirname
if args.indexdirname is None:
    indexdirname = os.path.dirname(args.filteredexpt)

if args.noiseless:
    GAIN = 1

if not os.path.exists(args.o) and rank == 0:
    os.makedirs(args.o)

MPI.COMM_WORLD.Barrier()

# load the bs7 default array
bs7_mil_ar = struct_fact_special.sfgen(WAVELEN_HIGH, "../sim/4bs7.pdb", yb_scatter_name="../sf/scanned_fp_fdp.tsv")
datasf_mil_ar = struct_fact_special.load_4bs7_sf()

# assert El_fnames
# if rank == 0:
#    print("I found %d fname" % len(El_fnames))
all_paths = []
all_Amats = []
all_bg_coeff = []
all_bg_success = []
odir = args.o

if args.bgname is not None:
    background = h5py.File(args.bgname, "r")['bigsim_d9114'][()]

writer = h5py.File(os.path.join(odir, "process_rank%d.h5" % rank), "w")

n_processed = 0
for i_shot in range(Nexper):

    if i_shot % size != rank:
        continue
    if rank == 0:
        print("Rank 0: Doing shot %d / %d" % (i_shot + 1, Nexper))

    # get the experiment stuffs
    Exper = El[i_shot]
    iset = Exper.imageset
    fpath = iset.get_path(0)
    if args.imgdirname is not None:
        fpath = fpath.split("/kaladin/")[1]
        fpath = os.path.join(args.imgdirname, fpath)
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
    mos_spread = float(h5["mos_spread"][()])
    Ncells_abc = tuple(map(int, h5["Ncells_abc"][()]))
    mos_doms = int(h5["mos_doms"][()])
    profile = h5["profile"][()]
    beamsize = float(h5["beamsize_mm"][()])
    exposure_s = float(h5["exposure_s"][()])
    spectrum = h5["spectrum"][()]
    total_flux = np.sum(spectrum)
    xtal_size = 0.0005  # h5["xtal_size_mm"][()]

    # <><><><><><><><><><><><><><
    # HERE WE WILL DO PREDICTIONS
    # <><><><><><><><><><><><><><
    # make a sad spectrum
    FLUX = [total_flux]
    # loading the beam  (this might have wrong energy)
    BEAM = Exper.beam
    if args.forcelambda is None:
        ave_wave = BEAM.get_wavelength()
    else:
        ave_wave = args.forcelambda
    energies = [parameters.ENERGY_CONV / ave_wave]

    # grab the detector
    # DET = Exper.detector
    DET = CSPAD
    detdist = abs(DET[0].get_origin()[-1])
    pixsize = DET[0].get_pixel_size()[0]
    fs_dim, ss_dim = DET[0].get_image_size()
    n_panels = len(DET)
    # grab the crystal
    crystal = Exper.crystal

    # Optionally use the ground truth crystal A matrix and detector for prediction (only possible with simulations)
    if args.usegt:
        crystal.set_A(h5["crystalA"][()])
        DET = deepcopy(CSPAD)  # and use the ground truth CSPAD

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
    # if args.miller is not None:
    #    if args.miller == "bs7":
    #        FF = [bs7_mil_ar]
    #    if args.miller == "datasf":
    #        FF = [datasf_mil_ar]

    #   <><><><><><><><><><><><><><><><><><><><>
    #   DO THE SIMULATION TO USE FOR PREDICTIONS
    #   <><><><><><><><><><><><><><><><><><><><>
    # choose a device Id for GPU
    device_Id = np.random.choice(range(n_gpu))
    # call the simulation helper
    simsAB = sim_utils.sim_colors(
        crystal, DET, BEAM, FF,
        energies,
        FLUX, pids=None, profile=profile, cuda=not args.nocuda, oversample=1,
        Ncells_abc=Ncells_abc, mos_dom=1, mos_spread=0,
        master_scale=1, recenter=True,
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
        fast, slow = DET[int(panel_id)].get_image_size()
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
    refl_strong_fname = os.path.join(indexdirname,
                                     "idx-" + os.path.basename(fpath.replace(".npz", "_strong.refl")))
    refls_strong = flex.reflection_table.from_file(refl_strong_fname)
    # make mask of all strong spot pixels..
    nfast, nslow = DET[0].get_image_size()
    img_shape = nslow, nfast  # numpy format
    # make a mask that tells me True if I am a background pixel
    is_bg_pixel = np.ones((n_panels, nslow, nfast), bool)
    # group the refls by panel ID
    refls_strong_perpan = prediction_utils.refls_by_panelname(refls_strong)
    for panel_id in refls_strong_perpan:
        fast, slow = DET[int(panel_id)].get_image_size()
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
    exper.imageset = iset  # Exper.imageset
    results = tilt_fit(  # TODO rewrite tilt fit so its infinitely more readable
        imgs=img_data, is_bg_pix=is_bg_pixel,
        delta_q=args.deltaq, photon_gain=GAIN, sigma_rdout=sigma_readout, zinger_zscore=args.Z,
        exper=exper, predicted_refls=refls_predict, sb_pad=args.sbpad)

    # do global background fit
    if rank==0:
        print("Estimating background!")
    bg_est_pix = np.hstack([I[bg].ravel() for I, bg in zip(BG_ESTIMATE, is_bg_pixel)])
    data_pix = np.hstack([I[bg].ravel() for I, bg in zip(img_data, is_bg_pixel)])
    t = time.time()
    out = minimize(min_panel, x0=[1], args=(data_pix, bg_est_pix), method='Nelder-Mead')
    if rank == 0:
        if out.success:
            print("background success: coef=%f took %f sec" % (out.x, time.time()-t))
        else:
            print("background failure")
    all_bg_coeff.append( out.x[0])
    all_bg_success.append(out.success)

    refls_predict, tilt_abc, error_in_tilt, I_Leslie99, varI_Leslie99 = results
    spot_snr = np.array(I_Leslie99) / np.sqrt(varI_Leslie99)
    spot_snr[np.isnan(spot_snr)] = -999  # sometimes variance is 0, leading to nan snr values..
    shoeboxes = refls_predict['shoebox']
    bboxes = np.vstack([list(sb.bbox)[:4] for sb in shoeboxes])
    bbox_panel_ids = np.array(refls_predict['panel'])
    Hi = np.vstack(refls_predict['miller_index'])
    did_i_index = np.array(refls_predict['id']) != -1  # refls that didnt index should be labeled with -1
    boundary_spot = np.array(refls_predict['boundary'])

    int_refl_path = os.path.join(indexdirname,
                                 "idx-" + os.path.basename(fpath).replace(".h5.npz", ".h5_integrated.refl"))
    _R_shot = flex.reflection_table.from_file(int_refl_path)
    _R_shot_strong = Rmaster.select(Rmaster['id'] == i_shot)

    for _pid in range(len(DET)):
        xstrong, ystrong = [], []
        x0, y0 = [], []
        x, y = [], []

        rpp = prediction_utils.refls_by_panelname(refls_predict)
        # get nanoBragg spot prediction positions
        if _pid in rpp:
            x, y, _ = map(lambda x: np.array(x) - 0.5, prediction_utils.xyz_from_refl(rpp[_pid]))

        # get strong spot pos
        _R_shot_strong_panel = _R_shot_strong.select(_R_shot_strong["panel"] == _pid)
        if len(_R_shot_strong_panel) > 0:
            xstrong, ystrong, _ = map(lambda x: np.array(x) - 0.5, prediction_utils.xyz_from_refl(_R_shot_strong_panel))

        # get stills process prediction positions
        _R_panel = _R_shot.select(_R_shot["panel"] == _pid)
        if len(_R_panel) > 0:
            x0, y0, _ = map(lambda x: np.array(x) - 0.5, prediction_utils.xyz_from_refl(_R_panel))

        if list(x0) and list(x):
            tree = cKDTree(list(zip(x0, y0)))
            res = tree.query_ball_point(list(zip(x, y)), r=0.9)
            for i, r in enumerate(res):
                miller_nano = rpp[_pid]["miller_index"][i]
                if miller_nano[0] == 0 and miller_nano[1] == 0 and miller_nano[2] == 0:
                    if rpp[_pid]["id"][i] != -1:  # if the spot actually indexed
                        raise ValueError("a predicted spot cannot have miller index 0,0,0")
                if not r:
                    continue
                if len(r) > 1:
                    continue
                i_near = r[0]
                miller_stills = _R_panel["miller_index"][i_near]
                if miller_stills != miller_nano:
                    # raise ValueError("nanoBragg and stills process give different miller index")
                    nmismatch += 1
                    print("Bad: %s, %s" % (str(miller_stills), str(miller_nano)))

    bragg_hi += list(map(tuple, Hi))
    stills_hi += list(_R_shot["miller_index"])
    bragg_hi = list(set(bragg_hi))
    stills_hi = list(set(stills_hi))

    nstills_tot = len(stills_hi)
    nbragg_tot = len(bragg_hi)
    nstrong_tot += len(_R_shot_strong)
    # FIXME why is 0,0,0 being predicted using nanoBragg code??
    if rank == 0:
        print(
            "RANK %d, SHOT %d : nstrong=%d (%d overall), nstills_pred=%d, (%d unique overall) nbragg_pred=%d (%d unique overall), nmismatch=%d"
            % (
            rank, i_shot, len(_R_shot_strong), nstrong_tot, len(_R_shot), nstills_tot, len(refls_predict), nbragg_tot,
            nmismatch))

    if has_mpi and args.showcompleteness:
        all_bragg_hi = comm.reduce(bragg_hi, MPI.SUM, root=0)
        all_stills_hi = comm.reduce(stills_hi, MPI.SUM, root=0)

    if rank == 0 and args.showcompleteness:
        all_bragg_hi = utils.map_hkl_list(all_bragg_hi)
        all_stills_hi = utils.map_hkl_list(all_stills_hi)
        all_bragg_hi = list(set(all_bragg_hi))
        all_stills_hi = list(set(all_stills_hi))
        bragg_mset = miller.set(symm, flex.miller_index(all_bragg_hi), anomalous_flag=True)
        bragg_mset.setup_binner(d_max=999, d_min=2, n_bins=10)
        print("nanoBragg predictions:\n<><><><><><><><><><><><>")
        bragg_mset.completeness(use_binning=True).show()

        stills_mset = miller.set(symm, flex.miller_index(all_stills_hi), anomalous_flag=True)
        stills_mset.setup_binner(d_max=999, d_min=2, n_bins=10)
        print("Stills process predictions:\n<><><><><><><><><><><><>")
        stills_mset.completeness(use_binning=True).show()

    # <><><><><><><><><><><><><><><>
    # DO THE SANITY PLOTS (OPTIONAL)
    # <><><><><><><><><><><><><><><>
    if rank == 0 and args.sanityplots:
        refls_predict_bypanel = prediction_utils.refls_by_panelname(refls_predict)
        pause = args.pause
        for panel_id in refls_predict_bypanel:
            panel_img = img_data[panel_id]
            m = panel_img.mean()
            s = panel_img.std()
            vmax = m + 4 * s
            vmin = m - s
            ax.clear()
            im = ax.imshow(panel_img, vmax=vmax, vmin=vmin)
            int_mask = np.zeros(panel_img.shape).astype(np.bool)
            bg_mask = np.zeros(panel_img.shape).astype(np.bool)

            for i_ref in range(len(refls_predict_bypanel[panel_id])):
                ref = refls_predict_bypanel[panel_id][i_ref]
                i1, i2, j1, j2, _, _ = ref['shoebox'].bbox
                rect = plt.Rectangle(xy=(i1, j1), width=i2 - i1, height=j2 - j1, fc='none', ec='Deeppink')
                plt.gca().add_patch(rect)
                mask = ref['shoebox'].mask.as_numpy_array()[0]
                int_mask[j1:j2, i1:i2] = np.logical_or(mask == 5, int_mask[j1:j2, i1:i2])
                bg_mask[j1:j2, i1:i2] = np.logical_or(mask == 19, bg_mask[j1:j2, i1:i2])
            plt.draw()
            plt.pause(pause)
            im.set_data(int_mask)
            plt.title("panel%d: integration mask" % panel_id)
            im.set_clim(0, 1)
            plt.draw()
            plt.pause(pause)
            im.set_data(bg_mask)
            plt.title("panel%d: background mask" % panel_id)
            im.set_clim(0, 1)
            plt.draw()
            plt.pause(pause)

    all_paths.append(fpath)
    all_Amats.append(crystal.get_A())
    if rank == 0:
        print("Rank0: writing")
    # save the output!
    writer.create_dataset("bboxes/shot%d" % n_processed, data=bboxes, dtype=np.int, compression="lzf")
    writer.create_dataset("tilt_abc/shot%d" % n_processed, data=tilt_abc, dtype=np.float32, compression="lzf")
    writer.create_dataset("tilt_error/shot%d" % n_processed, data=error_in_tilt, dtype=np.float32, compression="lzf")
    writer.create_dataset("SNR_Leslie99/shot%d" % n_processed, data=spot_snr, dtype=np.float32, compression="lzf")
    writer.create_dataset("I_Leslie99/shot%d" % n_processed, data=I_Leslie99, dtype=np.float32, compression="lzf")
    writer.create_dataset("varI_Leslie99/shot%d" % n_processed, data=varI_Leslie99, dtype=np.float32, compression="lzf")
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

writer.create_dataset("background_estimate", data=BG_ESTIMATE, compression="lzf")
writer.create_dataset("background_coefficients", data=all_bg_coeff, compression="lzf")
writer.create_dataset("background_fit_success", data=all_bg_success, compression="lzf")
writer.create_dataset("Amatrices", data=all_Amats, compression="lzf")
writer.create_dataset("h5_path", data=np.array(all_paths, dtype="S"), compression="lzf")
writer.close()
