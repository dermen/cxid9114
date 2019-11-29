#!/usr/bin/env libtbx.python

from argparse import ArgumentParser
from copy import deepcopy
from scipy.ndimage.morphology import binary_dilation

parser = ArgumentParser("Make prediction boxes")

parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--pearl", action="store_true")
parser.add_argument("--sz", default=5, type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--savefigdir", default=None, type=str)
parser.add_argument("--glob", type=str, required=True, help="experiment list glob")
parser.add_argument("--Z", type=float, default=2)
parser.add_argument("--dilate", default=1, type=int)
parser.add_argument("--bgname", type=str, default="../sim/bs7real_cspad.h5")
parser.add_argument("--defaultF", type=float, default=1e3)
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
args = parser.parse_args()

import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
import h5py
from cctbx import miller, sgtbx
from scitbx.array_family import flex
from cctbx.crystal import symmetry

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
    if args.plot is not None or args.plottilt is not None:
        import pylab as plt
        from mpl_toolkits.mplot3d import Axes3D

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
sz = args.sz  # expand for fittin background plane

if not os.path.exists(args.o) and rank == 0:
    os.makedirs(args.o)

MPI.COMM_WORLD.Barrier()

# load the bs7 default array
from cxid9114.sf import struct_fact_special
from cxid9114.parameters import WAVELEN_HIGH
bs7_mil_ar = struct_fact_special.sfgen(WAVELEN_HIGH, "../sim/4bs7.pdb", yb_scatter_name="../sf/scanned_fp_fdp.npz")
datasf_mil_ar = struct_fact_special.load_4bs7_sf()

assert El_fnames
if rank == 0:
    print("I found %d fname" % len(El_fnames))
all_paths = []
all_Amats = []
odir = args.o

background = h5py.File(args.bgname, "r")['bigsim_d9114'][()]
#background = h5py.File("../sim/boop_cspad.h5", "r")['bigsim_d9114'][()]

writer = h5py.File(os.path.join(odir, "process_rank%d.h5" % rank), "w")

n_processed = 0
for i_shot, (El_json, refl_pkl) in enumerate(zip(El_fnames, refl_fnames)):
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
    if args.pearl:  # debugs
        _fpath = fpath.replace("swirl", "pearl")
    else:
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

    # load reflections
    # Note this is used un order to make the strong spot mask to remove spots from background fits
    refls_data = utils.open_flex(refl_pkl)

    # make a sad spectrum
    # FIXME: set an actual miller array where the nonzero elements in GT are
    # set to a constant, that way we stay safe from predicting systematic absences
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

    sgi = sgtbx.space_group_info(args.symbol)
    # TODO: allow override of ucell
    symm = symmetry(unit_cell=crystal.get_unit_cell(), space_group_info=sgi)
    miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
    # NOTE does build_miller_set automatically expand to p1 ?
    # consider predicting with the ground truth here for a control
    Famp = flex.double(np.ones(len(miller_set.indices())) * args.defaultF)
    mil_ar = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()
    if args.miller is not None:
        if args.miller == "bs7":
            FF = [bs7_mil_ar]
        if args.miller == "datasf":
            FF = [datasf_mil_ar]

    detdist = abs(DET[0].get_origin()[-1])
    pixsize = DET[0].get_pixel_size()[0]
    fs_dim, ss_dim = DET[0].get_image_size()
    if args.usepredictions:
        beams = []
        device_Id = i_shot % n_gpu
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
                R = prediction_utils.refls_from_sims(simsAB[i_en], DET, beam, thresh=args.thresh)
            except:
                continue
            refls_at_colors.append(R)
            beams.append(beam)

        if not refls_at_colors:
            continue

        # this gets the integration shoeboxes, not to be confused with strong spot bound boxes
        Hi, bboxes, bbox_panel_ids, bbox_masks, patches, Pterms, Pterms_idx, integrated_Hi = prediction_utils.get_prediction_boxes(
            refls_at_colors,
            DET, beams,
            crystal, delta_q=delta_q, ret_Pvals=True,
            data=img_data, ret_patches=True, refls_data=refls_data, gain=GAIN,
            fc='none', ec='r')  # ret_patches=True, fc='none', ec='w')

        # dilate masks
        dilate_factor = args.dilate
        bbox_masks_dilated = []
        for mask in bbox_masks:
            bbox_masks_dilated.append(
                binary_dilation(mask, iterations=dilate_factor))

        if args.sanitycheck:
            refl_i_f = refl_pkl.replace("_strong.refl", "_indexed.refl")
            if not os.path.exists(refl_i_f):
                raise IOError("Reflection indexed file  %s does not exist!" % refl_i_f)
            R = utils.open_flex(refl_i_f)
            Rpp = prediction_utils.refls_by_panelname(R)
            for pid in Rpp:
                bb_on_panel = np.array(bboxes)[np.array(bbox_panel_ids) == pid]

                if args.plot is not None and rank == 0:
                    plt.figure(1)
                    plt.gcf().clear()
                    _dat = img_data[pid][img_data[pid] > 0]
                    m = _dat.mean()
                    s = _dat.std()
                    vmin = m-s
                    vmax = m+5*s
                    plt.imshow(img_data[pid], vmax=vmax, vmin=vmin, cmap='viridis')
                    nspots = len(bb_on_panel)
                    for i_spot in range(nspots):
                        x1, x2, y1, y2 = bb_on_panel[i_spot]
                        patch = plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fc='none', ec='r')
                        plt.gca().add_patch(patch)
                    plt.title("Panel=%d" % pid)

                    # get the ground truth background plane and plot
                    if args.savefigdir is not None:
                        plt.savefig(os.path.join(args.savefigdir, "_figure%d.png" % pid))
                    plt.draw()
                    plt.pause(args.plot)

                r_on_panel = Rpp[pid]
                x, y, _ = prediction_utils.xyz_from_refl(r_on_panel)
                x = np.array(x)-0.5
                y = np.array(y)-0.5
                Hi_on_panel = np.array(Hi)[np.array(bbox_panel_ids) == pid]

                for i_spot, (x1, x2, y1, y2) in enumerate(bb_on_panel):
                    inX = np.logical_and(x1 < x, x < x2)
                    inY = np.logical_and(y1 < y, y < y2)
                    in_bb = inX & inY
                    if not any(in_bb):
                        continue

                    pos = np.where(in_bb)[0]
                    for p in pos:
                        h_pred_nb = Hi_on_panel[i_spot]
                        h_pred_stills = r_on_panel[p]["miller_index"]
                        print "panel:", pid, h_pred_nb, h_pred_stills
            if args.savefigdir is not None:
                exit()
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

    # In addition to the integration mask, make a secondary mask of all strong spots

    nfast, nslow = DET[0].get_image_size()
    img_shape = nslow, nfast  # numpy format
    n_panels = len(DET)
    is_bg_pixel = np.ones((n_panels, nslow, nfast), bool)
    refls_data_perpan = prediction_utils.refls_by_panelname(refls_data)
    for panel_id in refls_data_perpan:
        fast, slow = DET[panel_id].get_image_size()
        mask = prediction_utils.strong_spot_mask_dials(
            refls_data_perpan[panel_id], (slow, fast),
            as_composite=True)
        mask = binary_dilation(mask, iterations=args.dilate)
        is_bg_pixel[panel_id] = ~mask

    if args.usepredictions:
        for i_bbox, (i1, i2, j1, j2) in enumerate(bboxes):
            integration_mask = bbox_masks_dilated[i_bbox]
            pid = bbox_panel_ids[i_bbox]
            bg = is_bg_pixel[pid, j1:j2, i1:i2]
            is_bg_pixel[pid, j1:j2, i1:i2] = ~np.logical_or(~bg, integration_mask)

    # TODO per panel strong spot mask
    # dxtbx detector size is (fast scan x slow scan)
    #nfast, nslow = DET[0].get_image_size()
    #img_shape = nslow, nfast  # numpy format
    #n_panels = len(DET)
    #is_bg_pixel = np.ones((n_panels, nslow, nfast), bool)
    #strong_panel_ids = refls_data["panel"]
    # mask the strong spots so they dont go into tilt plane fits
    #for i_refl, (x1, x2, y1, y2, _, _) in enumerate(refls_data['bbox']):
    #    i_panel = strong_panel_ids[i_refl]
    #    bb_ss = slice(y1, y2, 1)
    #    bb_fs = slice(x1, x2, 1)
    #    is_bg_pixel[i_panel, bb_ss, bb_fs] = False

    # load the images
    # TODO determine if direct loading of the image file is faster than using ExperimentList check_format=True
    #raw_data = iset.get_raw_data(0)
    #if isinstance(raw_data, tuple):
    #    assert len(raw_data) == len(DET)
    #    imgs = [flex_img.as_numpy_array() for flex_img in raw_data]
    #else:
    #    imgs = [raw_data.as_numpt_array()]

    imgs = img_data
    assert all([img.shape == (nslow, nfast) for img in imgs])

    # fit the background plane
    tilt_abc = []
    successes = []
    error_in_tilt = []
    for i_bbox, (_i1, _i2, _j1, _j2) in enumerate(bboxes):
        i1 = max(_i1-sz, 0)
        i2 = min(_i2+sz, fs_dim)

        j1 = max(_j1-sz, 0)
        j2 = min(_j2+sz, ss_dim)
        i_panel = bbox_panel_ids[i_bbox]
        shoebox_img = imgs[i_panel][j1:j2, i1:i2] / GAIN  # NOTE: gain is imortant here!
        bg = background[i_panel][j1:j2, i1:i2]
        # TODO: Consider moving the bg fitting to the instantiation of the Refiner..
        # TODO: ...to protect the case when bg planes are fit to images without gain correction
        shoebox_mask = is_bg_pixel[i_panel, j1:j2, i1:i2]
        #shoebox_mask2 = np.ones_like(shoebox_mask)
        #shoebox_mask2[sz:-sz, sz:-sz] = ~bbox_masks_dilated[i_bbox]
        #shoebox_mask_total = ~np.logical_or(~shoebox_mask2, ~shoebox_mask)
        try:
            tilt, bgmask, coeff, _, tilt_resid = tilting_plane(
                shoebox_img,
                mask=shoebox_mask,  # mask specifies which spots are bg pixels...
                zscore=args.Z,  # zscore is standard M.A.D. score for removing additional zingers from the fit
                spline=args.spline,
                return_resid=True)
            successes.append(True)
        except:
            successes.append(False)
        # note this fit should be EXACT, linear regression..
        tilt_abc.append((coeff[1], coeff[2], coeff[0]))  # store as fast-scan coeff, slow-scan coeff, offset coeff
        # NOTE try the residual instead of the ground truth as a filter
        #error_in_tilt.append(np.sqrt(np.mean((tilt-bg)**2)))
        nbg_pix = np.logical_and(~bgmask, shoebox_mask).sum()
        error_in_tilt.append(tilt_resid[0] / nbg_pix)
        if rank == 0 and args.debug:
            print("Bbox %d/%d, error in tilt=%.4f" % (i_bbox, len(bboxes), error_in_tilt[-1]))

        if args.plottilt is not None and rank == 0:
            if args.debug:
                shoebox_img_noiseless = imgs_noiseless[i_panel][j1:j2, i1:i2]

                Fig2 = plt.figure(2)
                ax3d = Fig2.add_subplot(111, projection="3d")
                plt.figure(2)
                ax3d.clear()
                YY, XX = np.indices(bg.shape)
                ax3d.plot_surface(XX, YY, tilt, cmap='gnuplot')
                ax3d.plot_surface(XX, YY, bg, cmap='cool')

                Fig3 = plt.figure(3)
                ax3 = Fig3.add_subplot(111, projection="3d")
                plt.figure(3)
                ax3.clear()
                ax3.plot_surface(XX, YY, tilt, cmap='gnuplot')
                ax3.plot_surface(XX, YY, shoebox_img, cmap='cool')

                plt.figure(1)
                plt.subplot(131)
                plt.title("tilt fit")
                plt.imshow(tilt)
                cl = plt.gca().images[0].get_clim()

                plt.subplot(132)
                plt.title("noiseless data")
                plt.imshow(shoebox_img_noiseless * shoebox_mask)
                plt.gca().images[0].set_clim(cl)

                plt.subplot(133)
                plt.title("data with dilated mask")
                plt.imshow(shoebox_img * shoebox_mask)
                plt.gca().images[0].set_clim(cl)
                plt.show()
                #plt.draw()
                #plt.pause(args.plottilt)

            else:
                plt.gcf().clear()
                plt.subplot(121)
                plt.imshow(tilt)
                plt.title("tilt fit")
                cl = plt.gca().images[0].get_clim()
                plt.subplot(122)
                plt.imshow(shoebox_img*shoebox_mask)
                plt.gca().images[0].set_clim(cl)
                plt.draw()
                plt.title("data with dilated mask")
                plt.pause(args.plottilt)

    assert all(successes)
    #bboxes = [b for i, b in enumerate(bboxes) if successes[i]]
    #Hi = [h for i, h in enumerate(Hi) if successes[i]]
    #bbox_panel_ids = [p for i, p in enumerate(bbox_panel_ids) if successes[i]]

    if not bboxes:
        continue
    all_paths.append(fpath)
    all_Amats.append(crystal.get_A())
    if rank == 0:
        print("Rank0: writing")
    writer.create_dataset("bboxes/shot%d" % n_processed, data=bboxes,  dtype=np.int, compression="lzf" )
    writer.create_dataset("tilt_abc/shot%d" % n_processed, data=tilt_abc,  dtype=np.float32, compression="lzf" )
    writer.create_dataset("tilt_rms/shot%d" % n_processed, data=error_in_tilt,  dtype=np.float32, compression="lzf" )
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

