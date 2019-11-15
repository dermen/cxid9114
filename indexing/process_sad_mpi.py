#!/usr/bin/env libtbx.python

from argparse import ArgumentParser
from copy import deepcopy

from IPython import embed

parser = ArgumentParser("Make prediction boxes")

parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--nrank", type=int, default=1)
parser.add_argument("--glob", type=str, required=True, help="experiment list glob")
parser.add_argument("-o", help='output directoty',  type=str, default='.')
parser.add_argument("--plot", action="store_true")
parser.add_argument("--usegt", action="store_true")
parser.add_argument("--usepredictions", action="store_true")
parser.add_argument("--show_params", action='store_true')
parser.add_argument("--forcelambda", type=float, default=None)
parser.add_argument("--noiseless", action="store_true")
args = parser.parse_args()

import numpy as np
if args.plot:
    import pylab as plt
import pandas
from dxtbx.model.experiment_list import ExperimentListFactory
import h5py

from cxid9114.sim import sim_utils
from cxid9114.geom.multi_panel import CSPAD


from cxid9114 import parameters, utils
from cxid9114.prediction import prediction_utils
from simtbx.diffBragg.utils import tilting_plane

import glob
import os

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

GAIN = 28
if args.noiseless:
    GAIN = 1
delta_q = 0.0475
sz = 5  # expand for fittin background plane

if not os.path.exists(args.o) and rank == 0:
    os.makedirs(args.o)

MPI.COMM_WORLD.Barrier()

assert El_fnames
if rank == 0:
    print("I found %d fname" % len(El_fnames))
all_paths = []
all_Amats = []
odir = args.o

writer = h5py.File(os.path.join(odir, "process_rank%d.h5" % rank), "w")

n_processed = 0
for i_shot, (El_json, refl_pkl) in enumerate(zip(El_fnames, refl_fnames)):
    #if i_shot < 2235:
    #    continue
    if i_shot % size != rank:
        continue
    if rank == 0:
        print("Rank 0: Doing shot %d / %d" % (i_shot + 1, len(El_fnames)))

    El = ExperimentListFactory.from_json_file(El_json, check_format=True)
    # El = ExperimentListFactory.from_json_file(El_json, check_format=False)

    iset = El.imagesets()[0]
    fpath = iset.get_path(0)
    # this is the file containing relevant simulation parameters..
    h5 = h5py.File(fpath.replace(".npz", ""), 'r')

    # get image pixels
    if args.noiseless:
        # load the image that has no noise
        noiseless_fpath = fpath.replace(".npz", ".noiseless.npz")
        assert os.path.exists(noiseless_fpath)
        img_data = np.load(noiseless_fpath)["img"]
    else:
        img_data = np.load(fpath)["img"]

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

    # load reflections
    # Note this is used un order to make the strong spot mask to remove spots from background fits
    refls_data = utils.open_flex(refl_pkl)

    # make a sad spectrum
    FF = [1e3]  # NOTE: not sure what to do here, we dont know the structure factor
    FLUX = [total_flux]

    # loading the beam  (this might have wrong energy)
    BEAM = El.beams()[0]
    if args.forcelambda is None:
        ave_wave = BEAM.get_wavelength()
    else:
        ave_wave = args.forcelambda
    energies = [parameters.ENERGY_CONV/ave_wave]
    DET = El.detectors()[0]  # load detector
    crystal = El.crystals()[0]  # load crystal
    if args.usegt:  # if true then use the ground truth crystal A matrix for prediction
        crystal.set_A(h5["crystalA"][()])
        DET = deepcopy(CSPAD)   # and use the ground truth CSPAD
    detdist = abs(DET[0].get_origin()[-1])
    pixsize = DET[0].get_pixel_size()[0]
    fs_dim, ss_dim = DET[0].get_image_size()
    if args.usepredictions:
        beams = []
        device_Id = rank % n_gpu
        simsAB = sim_utils.sim_colors(
            crystal, DET, BEAM, FF,
            energies,
            FLUX, pids=None, profile=profile, cuda=True, oversample=1,
            Ncells_abc=Ncells_abc, mos_dom=50, mos_spread=0.02,
            master_scale=1,
            exposure_s=exposure_s, beamsize_mm=beamsize, device_Id=device_Id,
            show_params=args.show_params, accumulate=False, crystal_size_mm=xtal_size)

        refls_at_colors = []
        for i_en, en in enumerate(energies):
            beam = deepcopy(BEAM)
            beam.set_wavelength(parameters.ENERGY_CONV/en)
            try:
                # NOTE: this is a multi panel refl table
                R = prediction_utils.refls_from_sims(simsAB[i_en], DET, beam, thresh=1e-2)
            except:
                continue
            refls_at_colors.append(R)
            beams.append(beam)

        if not refls_at_colors:
            continue

        # this gets the integration shoeboxes, not to be confused with strong spot bound boxes
        Hi, bboxes, bbox_panel_ids, patches, Pterms, Pterms_idx, integrated_Hi = prediction_utils.get_prediction_boxes(
            refls_at_colors,
            DET, beams,
            crystal, delta_q=delta_q, ret_Pvals=True,
            data=img_data, ret_patches=True, refls_data=refls_data, gain=GAIN,
            fc='none', ec='r')  # ret_patches=True, fc='none', ec='w')
    else:
        if not refl_indexed_fnames:
            raise ValueError("Need some refl_indexed_fnames, has size 0")
        # the indexed reflections table
        refl_idx = utils.open_flex(refl_indexed_fnames[i_shot])

        # specify panel ID
        bbox_panel_ids = refl_idx["panel"].as_numpy_array()

        # specify dimension of bound box (increases with resolution)
        Qmag = np.linalg.norm(np.array(refl_idx['rlp']), axis=1) * 2 * np.pi
        rad1 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag-delta_q*.5)*ave_wave/4/np.pi))
        rad2 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag+delta_q*.5)*ave_wave/4/np.pi))
        delrad = rad2-rad1
        # create the bound boxes
        bboxes = []
        for i_spot in range(len(refl_idx)):
            x_com, y_com, _ = refl_idx[i_spot]["xyzobs.px.value"]
            x_com += -0.5
            y_com += -0.5
            i1 = int(max(x_com - delrad[i_spot]/2., 0))
            i2 = int(min(x_com + delrad[i_spot]/2., fs_dim))
            j1 = int(max(y_com - delrad[i_spot]/2., 0))
            j2 = int(min(y_com + delrad[i_spot]/2., ss_dim))
            bboxes.append((i1, i2, j1, j2))   # i is fast scan, j is slow scan

        # load the indexes
        Hi = np.array(refl_idx["miller_index"])

    if len(Hi) != len(bbox_panel_ids):
        print("There is something funky! different number of panels and Hi")
        if has_mpi:
            comm.Abort()
        else:
            exit()

    # TODO per panel strong spot mask
    # dxtbx detector size is (fast scan x slow scan)
    nfast, nslow = DET[0].get_image_size()
    img_shape = nslow, nfast  # numpy format
    n_panels = len(DET)
    is_bg_pixel = np.ones((n_panels, nslow, nfast), bool)
    strong_panel_ids = refls_data["panel"]
    # mask the strong spots so they dont go into tilt plane fits
    for i_refl, (x1, x2, y1, y2, _, _) in enumerate(refls_data['bbox']):
        i_panel = strong_panel_ids[i_refl]
        bb_ss = slice(y1, y2, 1)
        bb_fs = slice(x1, x2, 1)
        is_bg_pixel[i_panel, bb_ss, bb_fs] = False

    # load the images
    # TODO determine if direct loading of the image file is faster than using ExperimentList check_format=True
    raw_data = iset.get_raw_data(0)
    if isinstance(raw_data, tuple):
        assert len(raw_data) == len(DET)
        imgs = [flex_img.as_numpy_array() for flex_img in raw_data]
    else:
        imgs = [raw_data.as_numpt_array()]
    assert all([img.shape == (nslow, nfast) for img in imgs])

    # fit the background plane
    tilt_abc = []
    successes = []
    for i_bbox, (_i1, _i2, _j1, _j2) in enumerate(bboxes):
        i1 = max(_i1-sz, 0)
        i2 = min(_i2+sz, fs_dim)

        j1 = max(_j1-sz, 0)
        j2 = min(_j2+sz, ss_dim)
        i_panel = bbox_panel_ids[i_bbox]
        shoebox_img = imgs[i_panel][j1:j2, i1:i2] / GAIN  # NOTE: gain is imortant here!
        # TODO: Consider moving the bg fitting to the instantiation of the Refiner..
        # TODO: ...to protect the case when bg planes are fit to images without gain correction
        shoebox_mask = is_bg_pixel[i_panel, j1:j2, i1:i2]
        try:
            tilt, bgmask, coeff, _ = tilting_plane(
                shoebox_img,
                mask=shoebox_mask,  # mask specifies which spots are bg pixels...
                zscore=2)  # zscore is standard M.A.D. score for removing additional zingers from the fit
            successes.append(True)
        except:
            successes.append(False)
        # note this fit should be EXACT, linear regression..
        tilt_abc.append((coeff[1], coeff[2], coeff[0]))  # store as fast-scan coeff, slow-scan coeff, offset coeff

    assert all(successes)
    bboxes = [b for i, b in enumerate(bboxes) if successes[i]]
    Hi = [h for i, h in enumerate(Hi) if successes[i]]
    bbox_panel_ids = [p for i, p in enumerate(bbox_panel_ids) if successes[i]]

    if not bboxes:
        continue
    all_paths.append(fpath)
    all_Amats.append(crystal.get_A())
    if rank == 0:
        print("Rank0: writing")
    writer.create_dataset("bboxes/shot%d" % n_processed, data=bboxes,  dtype=np.int, compression="lzf" )
    writer.create_dataset("tilt_abc/shot%d" % n_processed, data=tilt_abc,  dtype=np.float32, compression="lzf" )
    writer.create_dataset("Hi/shot%d" % n_processed, data=Hi, dtype=np.int, compression="lzf")
    writer.create_dataset("panel_ids/shot%d" % n_processed, data=bbox_panel_ids, dtype=np.int, compression="lzf")
    # add the default peak selection flags (default is all True, so select all peaks for refinement)
    keepers = np.ones(len(bboxes)).astype(np.bool)
    writer.create_dataset("bboxes/keepers%d" % n_processed, data=keepers, dtype=np.bool, compression="lzf")

    #writer.create_dataset("bg_pixel_mask/shot%d" % n_processed, data=is_bg_pixel, dtype=bool, compression="lzf")
    #sg96 = sgtbx.space_group(" P 4nw 2abw")
    ## ground truth structure factors ?
    #FA = load_4bs7_sf()
    #HA = tuple([hkl for hkl in FA.indices()])
    #HA_val_map = {h: data for h, data in zip(FA.indices(), FA.data())}
    #Hmaps = [HA_val_map]

    #def get_val_at_hkl(hkl, val_map):
    #    poss_equivs = [i.h() for i in
    #                   miller.sym_equiv_indices(sg96, hkl).indices()]
    #    in_map = False
    #    for hkl2 in poss_equivs:
    #        if hkl2 in val_map:  # fast lookup
    #            in_map = True
    #            break
    #    if in_map:
    #        return hkl2, val_map[hkl2]
    #    else:
    #        return (None, None, None), -1

    #K = FF[0] ** 2 * FLUX[0] * exposure_s
    #LA = FLUX[0] * exposure_s
    #L_at_color = [LA]

    #Nh = len(Hi)
    #rhs = []
    #lhs = []
    #all_H2 = []
    #all_PA = []
    #all_PB = []
    #all_FA = []
    #all_FB = []
    #for i in range(Nh):
    #    HKL = Hi[i]
    #    yobs = integrated_Hi[i]
    #    Pvals = Pterms[i]
    #    ycalc = 0

    #    # NOTE only support for Pvals len==1
    #    assert len(Pvals) == 1
    #    for i_P, P in enumerate(Pvals):
    #        L = L_at_color[i_P]
    #        H2, F = get_val_at_hkl(HKL, Hmaps[i_P])
    #        if i_P == 0:
    #            all_FA.append(F)
    #        #else:
    #        #    all_FB.append(F)

    #        ycalc += L * P * abs(F) ** 2 / K
    #    all_PA.append(Pvals[0])
    #    #all_PB.append(Pvals[1])
    #    all_H2.append(H2)
    #    rhs.append(ycalc)
    #    lhs.append(yobs)
    #all_PB = [0]*len(all_PA)
    #all_FB = [0]*len(all_FA)
    #df = pandas.DataFrame({"rhs": rhs, "lhs": lhs,
    #                       "PA": all_PA, "PB": all_PB, "FA": all_FA,
    #                       "FB": all_FB})

    #df["run"] = rank
    #df["shot_idx"] = i_shot
    #df['gain'] = 1

    #df['LA'] = LA
    #df['LB'] = 0
    #df['K'] = K

    #h, k, l = zip(*all_H2)
    #df['h2'] = h
    #df['k2'] = k
    #df['l2'] = l

    #pklname = fpath.replace(".h5.npz", ".pdpkl")
    #df.to_pickle(pklname)

    #if args.plot and rank == 0:
    #    print("PLOT")

    #    plt.plot(df.lhs, df.rhs, '.')
    #    plt.show()
    #print("DonDonee")

    if rank == 0:
        print("Rank0: Done writing")
    n_processed += 1

writer.create_dataset("Amatrices", data=all_Amats, compression="lzf")
writer.create_dataset("h5_path", data=all_paths, compression="lzf")
writer.close()

