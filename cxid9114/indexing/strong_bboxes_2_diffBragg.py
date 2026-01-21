#!/usr/bin/env libtbx.python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser("Make strong boxes", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--savefitsel", action="store_true")
parser.add_argument("--mono", action="store_true")
parser.add_argument("--keepbelowzero", action="store_true")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPU machines connected to current host")
parser.add_argument("--nocuda", action="store_true")
parser.add_argument("--flat", action="store_true")
parser.add_argument("--showcompleteness", action="store_true")
parser.add_argument("--savefigdir", default=None, type=str)
parser.add_argument("--filteredexpt", type=str, required=True, help="filtered combined experiment file")
parser.add_argument("--Z", type=float, default=2, help="zinger median absolute deviation Zscore")
parser.add_argument("--deltaq", type=float, default=0.07, help="reciprocal space width of bound box")
parser.add_argument("--dilate", default=1, type=int, help="factor by which to dilate the integration mask")
parser.add_argument("--defaultF", type=float, default=1e3, help="for prediction simulation use this value at every Fhkl")
parser.add_argument("--thresh", type=float, default=1e-2, help="simulated pixels above this value will be used to form the integration mask")
parser.add_argument("--o", help='output directoty', type=str, default='.')
parser.add_argument("--tag", help='output tag', type=str, default='boop')
parser.add_argument("--Nmax", help='max num exper to process', type=int, default=-1)
parser.add_argument("--show_params", action='store_true')
parser.add_argument("--imgdirname", type=str, default=None)
parser.add_argument("--indexdirname", type=str, default=None)
parser.add_argument("--symbol", default="P43212", type=str, help="space group symbol")
parser.add_argument("--sanityplots", action='store_true', help="whether to display plots for visual verification")
parser.add_argument("--pause", type=float, default=0.5, help="pause interval in seconds between consecutive plots")
parser.add_argument("--gain", type=float, default=9.481, help="value for adu per photon")
parser.add_argument("--readout", type=float, default=13.02)
parser.add_argument("--maskfile", type=str, default=None, help="path to an hdf5 file with `trusted_pixels` as a dataset name,\n data should be same shape as raw data (Num_panels, slowdim, fastdim), and dtype bool")
args = parser.parse_args()

#
# ave_sigma_r = (12.40503397943359+6.359566111586826 + 20.30000889831704)/3. = 13.02

GAIN = args.gain
sigma_readout = args.readout
bb = 8  # bound box half-width
import h5py


import glob
import os
from copy import deepcopy
import numpy as np
import h5py
from scipy.ndimage.morphology import binary_dilation
from IPython import embed
from scipy.spatial import cKDTree
from dxtbx.model.experiment_list import ExperimentListFactory, Experiment
from cctbx import miller, sgtbx
from dials.array_family import flex
from cctbx.crystal import symmetry
from cxid9114.sim import sim_utils
from cxid9114.geom.multi_panel import CSPAD
from cxid9114 import parameters, utils
from cxid9114.prediction import prediction_utils
from tilt_fit.tilt_fit import tilt_fit, TiltPlanes
import pandas
import time

mask_filename = '/global/cfs/cdirs/m3562/der/trusted_pixels.hdf5'
if args.maskfile is not None:
    mask_filename = args.maskfile
trusted_pixels = h5py.File(mask_filename, 'r')["trusted_pixels"][()]
badpixel_map = np.logical_not(trusted_pixels)

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

# Load in the reflection tables and experiment lists
from dxtbx.model import ExperimentList
from cxid9114.parameters import ENERGY_CONV
print ("Reading in the files")
El = ExperimentListFactory.from_json_file(args.filteredexpt, check_format=False)
if args.Nmax is not None:
    El = El[:args.Nmax]
Nexper = len(El)

DET = El.detectors()[0] 
if args.flat:
    det_El = ExperimentListFactory.from_json_file("flat_swiss.expt", check_format=False)
    DET = det_El.detectors()[0]
    assert DET[0].get_thickness()==0
    assert DET[0].get_mu()==0

Rmaster = flex.reflection_table.from_file(args.filteredexpt.replace(".expt", ".refl"))
print ("Read in %d experiments" % Nexper)
# get the original indexing directory name
indexdirname = args.indexdirname
if args.indexdirname is None:
    indexdirname = os.path.dirname(args.filteredexpt)

if not os.path.exists(args.o) and rank == 0:
    os.makedirs(args.o)

MPI.COMM_WORLD.Barrier()

all_master_paths = []
all_master_indices = []
all_Amats = []
odir = args.o

if not os.path.exists(args.o) and rank == 0:
    os.makedirs(args.o)

#writer = h5py.File(os.path.join(odir, "strong_process_rank%d.h5" % rank), "w")

### LOAD ALL POSSIBLE MASTER FILE PATHS
import dxtbx
loaders = {}
print("Loading all of the paths")
for i_e, Exper in enumerate(El):
    if i_e % 50==0:
        print("Loading %d / %d" % (i_e+1, Nexper))
    path = Exper.imageset.get_path(0)
    if path not in loaders:
        loaders[path] = dxtbx.load(path)


def fit_background_and_snr(refls, imgs, is_bg_pix, experiment, outlier_z=4,  bad_pix=None, GAIN=1,
                           sigma_readout=3, keepbelowzero=False, savefitsel=False, verbose=False):
    refls = TiltPlanes.prep_relfs_for_tiltalization(refls, exper=experiment)
    tiltnation = TiltPlanes(panel_imgs=imgs, panel_bg_masks=is_bg_pix, panel_badpix_masks=bad_pix)
    tiltnation.check_if_refls_are_formatted_for_this_class(refls)
    tiltnation.make_quick_bad_pixel_proximity_checker(refls)
    tiltnation.sigma_rdout = sigma_readout
    tiltnation.adu_per_photon = GAIN
    tiltnation.delta_Q = 0.06
    tiltnation.zinger_zscore = outlier_z
    tiltnation.pixsize_mm = experiment.detector[0].get_pixel_size()[0]
    tiltnation.ave_wavelength_A = experiment.beam.get_wavelength()
    tiltnation.detdist_mm = experiment.detector[0].get_distance()
    tiltnation.min_background_pix_for_fit = 10
    tiltnation.min_dist_to_bad_pix = 7
    tiltnation.verbose = verbose

    bboxes = []
    tilt_abc = []
    error_in_tilt = []
    I_Leslie99 = []
    varI_Leslie99 = []
    did_i_index = []
    boundary_spot = []
    bbox_panel_ids = []
    Hi = []
    indexed_Hi = []
    selected_ref_idx = []
    all_reso = []
    all_fit_sel = []
    all_below_zero = []
    nref = len(refls)
    for i_r in range(nref):
        ref = refls[i_r]
        mil_idx = [int(hi) for hi in ref["miller_index"]]

        if mil_idx == [0, 0, 0]:
            continue

        if mil_idx in indexed_Hi:
            if verbose:
                print("already indexed, this split across two panels!")
            continue

        result = tiltnation.integrate_shoebox(ref)
        if result is None:
            continue
        shoebox_roi, coefs, variance_matrix, Isum, varIsum, below_zero_flag, fit_sel = result
        if below_zero_flag and not keepbelowzero:
            if verbose:
                print("Tilt plane dips below 0!")
            continue

        i1, i2, j1, j2 = shoebox_roi
        pid = ref['panel']
        badpix_roi = bad_pix[pid][j1:j2, i1:i2]
        if np.any(badpix_roi):
            if verbose:
                print("bad pixel in the ROI")
            continue

        all_below_zero.append(below_zero_flag)
        bboxes.append(shoebox_roi)
        tilt_abc.append(coefs)
        error_in_tilt.append(np.diag(variance_matrix).sum())
        I_Leslie99.append(Isum)
        varI_Leslie99.append(varIsum)
        bbox_panel_ids.append(int(ref["panel"]))
        Hi.append(mil_idx)
        did_i_index.append(True)
        if savefitsel:
            all_fit_sel.append(fit_sel)
        x1, x2, y1, y2 = shoebox_roi
        if x1 == 0 or y1 == 0 or x2 == fs_dim or y2 == ss_dim:
            boundary_spot.append(True)
        else:
            boundary_spot.append(False)
        indexed_Hi.append(mil_idx)
        selected_ref_idx.append(i_r)
        reso = 1. / np.linalg.norm(ref['rlp'])
        all_reso.append(reso)

    #chosen_selection = flex.bool([i in selected_ref_idx for i in range(nref)])
    #refls_predict = refls_predict.select(chosen_selection)
    spot_snr = np.array(I_Leslie99) / np.sqrt(varI_Leslie99)
    spot_snr[np.isnan(spot_snr)] = -999

    return spot_snr, all_reso, bboxes, indexed_Hi, Hi, selected_ref_idx, bbox_panel_ids, \
           I_Leslie99, varI_Leslie99, did_i_index, tilt_abc, error_in_tilt, \
           all_below_zero, boundary_spot, all_fit_sel


writer = h5py.File(os.path.join(odir, "process_rank%d.h5" % rank), "w")

#shot_data = []
n_processed = 0
for i_shot in range(Nexper):
   
    if i_shot % size != rank:
        continue
    if rank == 0:
        print("Rank %d: Doing shot %d / %d" % (rank, i_shot + 1, Nexper))

    # get the experiment stuffs
    Exper = El[i_shot] 
    BEAM = Exper.beam
    iset = Exper.imageset
    filepath = iset.get_path(0)
    assert len(iset)==1
    fidx = iset.indices()[0]
    
    img_data = np.array([panel.as_numpy_array() for panel in loaders[filepath].get_raw_data(fidx)])
    sdim, fdim = img_data[0].shape
    # get simulation parameters
    Ncells_abc = 10,10,10 #60, 60, 60 
    profile = "gauss"
    beamsize_mm = 0.001
    exposure_s = 1
    fluences = BEAM.get_spectrum_weights().as_numpy_array()
    total_flux = 1e12
    xtal_size = 0.0005  

    # <><><><><><><><><><><><><><
    # HERE WE WILL DO PREDICTIONS
    # <><><><><><><><><><><><><><
    energies = BEAM.get_spectrum_energies().as_numpy_array()

    # bin the spectrum
    nbins = 100
    energy_bins = np.linspace(energies.min()-1e-6, energies.max()+1e-6, nbins+1) 
    fluences = np.histogram(energies, bins=energy_bins, weights=fluences)[0]
    energies = .5*(energy_bins[:-1] + energy_bins[1:]) 
   
    # only simulate if significantly above the baselein (TODO make more accurate)
    cutoff = np.median(fluences) * 0.8
    is_finite = fluences > cutoff
    fluences = fluences[is_finite]
    fluences /= fluences.sum()
    fluences *= total_flux
    energies = energies[is_finite]

    # mono sim
    if args.mono:
        ave_en = ENERGY_CONV / BEAM.get_wavelength()
        #sig = 8
        #energies = np.linspace(ave_en-50 ,ave_en+50, 100)
        #I = np.exp( -(ave_en-energies)**2 / 2/sig/sig)
        #I /= I.sum()
        #I *= total_flux
        #fluences = I
        energies = [ave_en]
        fluences = [total_flux]

    if args.sanityplots:
        ax.clear()
        ax.plot( energies, fluences)
        plt.draw()
        plt.pause(args.pause)


    # grab the detector datas
    detdist = abs(DET[0].get_origin()[-1])
    pixsize = DET[0].get_pixel_size()[0]
    fs_dim, ss_dim = DET[0].get_image_size()
    n_panels = len(DET)
    # grab the crystal
    crystal = Exper.crystal

    # make the miller array to be used for prediction
    sgi = sgtbx.space_group_info(args.symbol)
    symm = symmetry(unit_cell=crystal.get_unit_cell(), space_group_info=sgi)
    miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
    Famp = flex.double(np.ones(len(miller_set.indices())) * args.defaultF)
    mil_ar = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()
    FF = [mil_ar] + [None]*(len(energies)-1)

    # SELECT THE STRONG SPOT ROI
    exper_refls_strong = Rmaster.select(Rmaster['id']==i_shot)
    panel_ids = exper_refls_strong["panel"]
    panels_with_spots = set(panel_ids)
    alist_panels = list(panels_with_spots) 
    #alist_panels = list(range(64,72))
    panels_with_spots = [i for i in panels_with_spots if i in alist_panels]
    rois_perpanel = {}
    panel_keys = {}
    counts_perpanel = {}
    refl_ids = {}
    for ii, pid in enumerate(panels_with_spots):
        rois_perpanel[ii] = []
        refl_ids[ii] = []
        counts_perpanel[ii] = np.zeros((sdim, fdim))
        panel_keys[pid] = ii

    centroid_x, centroid_y, _ = map(lambda x: np.array(x)-0.5, prediction_utils.xyz_from_refl(exper_refls_strong))
    for i_ref, (i,j, pid) in enumerate(zip(centroid_x, centroid_y, panel_ids)):
        if pid not in alist_panels:
            continue
        i1 = int(max(0, i-bb))
        i2 = int(min(fdim, i+bb))
        j1 = int(max(0, j-bb))
        j2 = int(min(sdim, j+bb))
        roi = (i1, i2), (j1, j2)
        ii = panel_keys[pid]
        rois_perpanel[ii].append(roi)
        counts_perpanel[ii][j1:j2, i1:i2] += 1
        refl_ids[ii].append(i_ref)

    # <><><><><><><><><>
    # DO THE SIMULATION
    # <><><><><><><><><>
    # choose a device Id for GPU
    device_Id = np.random.choice(range(n_gpu))
    # call the simulation helper
    t = time.time()
    simsAB = sim_utils.sim_colors(
        crystal, DET, BEAM, FF,
        energies, fluences, pids=panels_with_spots, 
        profile=profile, cuda=not args.nocuda, oversample=1,
        Ncells_abc=Ncells_abc, mos_dom=1, mos_spread=0,
        master_scale=1, recenter=True,time_panels=False,
        roi_pp=rois_perpanel, counts_pp=counts_perpanel,
        exposure_s=exposure_s, beamsize_mm=beamsize_mm, device_Id=device_Id,
        show_params=args.show_params, accumulate=True, crystal_size_mm=xtal_size)
    tsim = time.time()-t

    datas, sims, i_refs, cents, cents_mm  = [],[],[],[],[]
    bboxes = []
    for pid in range(len(DET)):
        sim_panel = simsAB[pid]
        if sim_panel is None:
            continue
        panel_key = panel_keys[pid]
        rois = rois_perpanel[panel_key]
        for i_roi, ((i1,i2),(j1,j2)) in enumerate(rois):
            roi_data = img_data[pid][j1:j2, i1:i2]
            roi_sim = sim_panel[j1:j2, i1:i2]
            Ivals = roi_sim.ravel()
            Isum = Ivals.sum()
            if Isum == 0:
                continue
            Y, X = np.indices(roi_sim.shape)
            yvals = Y.ravel()
            xvals = X.ravel()
            xcom = (xvals*Ivals).sum()/Isum
            ycom = (yvals*Ivals).sum()/Isum
            xcom = xcom + i1 + .5
            ycom = ycom + j1 + .5
            centroid = xcom, ycom
            cents.append (centroid)
            centroid_mm = DET[pid].pixel_to_millimeter(centroid)
            cents_mm.append(centroid_mm)
            datas.append(roi_data)
            sims.append(roi_sim)
            i_ref = refl_ids[panel_key][i_roi]
            i_refs.append(i_ref)
            bboxes.append((i1, i2, j1, j2))

    is_bg_pix = np.zeros(img_data.shape, bool)
    thresh = 1
    for i, (i_ref, sim) in enumerate(zip(i_refs, sims)):
        i1, i2, j1, j2 = bboxes[i]
        panel_id = exper_refls_strong[i_ref]['panel']
        is_bg_pix[panel_id, j1:j2, i1:i2] = sim < thresh

    out = fit_background_and_snr(exper_refls_strong, img_data, is_bg_pix, Exper,
                                 GAIN=GAIN, savefitsel=args.savefitsel,
                                 bad_pix=badpixel_map, sigma_readout=args.readout)

    spot_snr, spot_reso, spot_bboxes, indexed_Hi, Hi, selected_ref_idx, bbox_panel_ids, \
        I_Leslie99, varI_Leslie99, did_i_index, tilt_abc, error_in_tilt, \
        below_zero_flags, boundary_spot, fit_selections = out
    if len(spot_snr) == 0:
        print("Rank %d, no reflections to save" % rank)
        continue

    #shot_data += list(zip([i_shot]*len(cents), cents, cents_mm, i_refs))

    ### SAVE SHIT
    all_master_paths.append(filepath)
    all_master_indices.append(fidx)
    all_Amats.append(crystal.get_A())
    if rank == 0:
        print("Rank%d: writing" % rank)
    # save the output!
    writer.create_dataset("bboxes/shot%d" % n_processed, data=spot_bboxes, dtype=np.int, compression="lzf")
    writer.create_dataset("tilt_abc/shot%d" % n_processed, data=tilt_abc, dtype=np.float32, compression="lzf")
    writer.create_dataset("tilt_error/shot%d" % n_processed, data=error_in_tilt, dtype=np.float32, compression="lzf")
    writer.create_dataset("SNR_Leslie99/shot%d" % n_processed, data=spot_snr, dtype=np.float32, compression="lzf")
    writer.create_dataset("resolution/shot%d" % n_processed, data=spot_reso, dtype=np.float32, compression="lzf")
    writer.create_dataset("I_Leslie99/shot%d" % n_processed, data=I_Leslie99, dtype=np.float32, compression="lzf")
    writer.create_dataset("varI_Leslie99/shot%d" % n_processed, data=varI_Leslie99, dtype=np.float32, compression="lzf")
    writer.create_dataset("Hi/shot%d" % n_processed, data=Hi, dtype=np.int, compression="lzf")
    writer.create_dataset("indexed_flag/shot%d" % n_processed, data=did_i_index, dtype=np.int, compression="lzf")
    writer.create_dataset("is_on_boundary/shot%d" % n_processed, data=boundary_spot, dtype=np.bool, compression="lzf")
    writer.create_dataset("panel_ids/shot%d" % n_processed, data=bbox_panel_ids, dtype=np.int, compression="lzf")
    if args.keepbelowzero:
        writer.create_dataset("below_zero/shot%d" % n_processed, data=below_zero_flags, dtype=np.bool, compression="lzf")
    if args.savefitsel:
        writer.create_dataset("fit_selection/shot%d" % n_processed, data=fit_selections, dtype=np.bool,
                              compression="lzf")

    # add the default peak selection flags (default is all True, so select all peaks for refinement)
    keepers = np.ones(len(spot_bboxes)).astype(np.bool)
    writer.create_dataset("bboxes/keepers%d" % n_processed, data=keepers, dtype=np.bool, compression="lzf")

    nref = len(exper_refls_strong)
    nref_save = len(spot_bboxes)
    frac_saved = float(nref_save) / nref * 100
    print("Rank %d: shot %d / %d  has %d refls, saving %d (%.1f %%) TOok %f seconds to simulate %d panels "
          % (rank, i_shot+1, Nexper, nref, nref_save, frac_saved, tsim, len(panels_with_spots)), flush=True)

    n_processed += 1


writer.create_dataset("Amatrices", data=all_Amats, compression="lzf")
writer.create_dataset("h5_path", data=np.array(all_master_paths, dtype="S"), compression="lzf")
writer.create_dataset("master_file_indices", data=all_master_indices, compression="lzf")
writer.close()

