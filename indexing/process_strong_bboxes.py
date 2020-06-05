#!/usr/bin/env libtbx.python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser("Make strong boxes", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--savefitsel", action="store_true")
parser.add_argument("--keepbelowzero", action="store_true")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPU machines connected to current host")
parser.add_argument("--nocuda", action="store_true")
parser.add_argument("--showcompleteness", action="store_true")
parser.add_argument("--savefigdir", default=None, type=str)
parser.add_argument("--filteredexpt", type=str, required=True, help="filtered combined experiment file")
parser.add_argument("--Z", type=float, default=2, help="zinger median absolute deviation Zscore")
parser.add_argument("--deltaq", type=float, default=0.07, help="reciprocal space width of bound box")
parser.add_argument("--dilate", default=1, type=int, help="factor by which to dilate the integration mask")
parser.add_argument("--defaultF", type=float, default=1e3, help="for prediction simulation use this value at every Fhkl")
parser.add_argument("--thresh", type=float, default=1e-2, help="simulated pixels above this value will be used to form the integration mask")
parser.add_argument("--o", help='output directoty', type=str, default='.')
parser.add_argument("--show_params", action='store_true')
parser.add_argument("--imgdirname", type=str, default=None)
parser.add_argument("--indexdirname", type=str, default=None)
parser.add_argument("--symbol", default="P43212", type=str, help="space group symbol")
parser.add_argument("--sanityplots", action='store_true', help="whether to display plots for visual verification")
parser.add_argument("--pause", type=float, default=0.5, help="pause interval in seconds between consecutive plots")
parser.add_argument("--gain", type=float, default=1, help="value for adu per photon")
parser.add_argument("--readout", type=float, default=3)
args = parser.parse_args()

GAIN = args.gain
sigma_readout = args.readout
bb = 8  # bound box shalf width

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
from cxid9114.sf import struct_fact_special
from cxid9114.parameters import WAVELEN_HIGH
from tilt_fit.tilt_fit import tilt_fit, TiltPlanes

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
print ("Reading in the files")
El = ExperimentListFactory.from_json_file(args.filteredexpt, check_format=False)
Nexper = len(El)

DET = El.detectors()[0] 
Rmaster = flex.reflection_table.from_file(args.filteredexpt.replace(".expt", ".refl"))
print ("Read in %d experiments" % Nexper)
# get the original indexing directory name
indexdirname = args.indexdirname
if args.indexdirname is None:
    indexdirname = os.path.dirname(args.filteredexpt)

if not os.path.exists(args.o) and rank == 0:
    os.makedirs(args.o)

MPI.COMM_WORLD.Barrier()

# load the bs7 default array
all_paths = []
all_Amats = []
odir = args.o

writer = h5py.File(os.path.join(odir, "strong_process_rank%d.h5" % rank), "w")

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

n_processed = 0
for i_shot in range(Nexper):
    
    if i_shot % size != rank:
        continue
    if rank==0:
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
    Ncells_abc = 60, 60, 60 
    profile = "gauss"
    beamsize_mm = 0.001
    exposure_s = 1
    fluences = BEAM.get_spectrum_weights().as_numpy_array()
    total_flux = 1e12
    xtal_size = 0.0005  

    # <><><><><><><><><><><><><><
    # HERE WE WILL DO PREDICTIONS
    # <><><><><><><><><><><><><><
    # make a sad spectrum
    # loading the beam  (this might have wrong energy)
    energies = BEAM.get_spectrum_energies().as_numpy_array()

    nbins = 60
    energy_bins = np.linspace(energies.min()-1e-6, energies.max()+1e-6, nbins+1) 
    fluences = np.histogram(energies, bins=energy_bins, weights=fluences)[0]
    energies = .5*(energy_bins[:-1] + energy_bins[1:]) 
    
    cutoff = np.median(fluences) * 0.5
    is_finite = fluences > cutoff
    
    fluences = fluences[is_finite]
    fluences /= fluences.sum()
    fluences *= total_flux
    energies = energies[is_finite]

    if args.sanityplots:
        ax.clear()
        ax.plot( energies, fluences)
        plt.draw()
        plt.pause(args.pause)

    # mono sim
    #energies = [(energies*fluences).sum() / fluences.sum()]
    #fluences = [total_flux]

    # grab the detector
    detdist = abs(DET[0].get_origin()[-1])
    pixsize = DET[0].get_pixel_size()[0]
    fs_dim, ss_dim = DET[0].get_image_size()
    n_panels = len(DET)
    # grab the crystal
    crystal = Exper.crystal

    # make the miller array to be used with prediction
    sgi = sgtbx.space_group_info(args.symbol)
    # TODO: allow override of ucell
    symm = symmetry(unit_cell=crystal.get_unit_cell(), space_group_info=sgi)
    miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
    # NOTE does build_miller_set automatically expand to p1 ? Does it obey systematic absences ?
    # Note how to handle sys absences here ?
    Famp = flex.double(np.ones(len(miller_set.indices())) * args.defaultF)
    mil_ar = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()
    FF = [mil_ar] + [None]*(len(energies)-1)

    # SELECT THE STRONG SPOT ROI
    exper_refls_strong = Rmaster.select(Rmaster['id']==i_shot)
    panel_ids = exper_refls_strong["panel"]
    panels_with_spots = set(panel_ids)
    rois_perpanel = {}
    panel_keys = {}
    counts_perpanel = {}
    for ii, pid in enumerate(panels_with_spots):
        rois_perpanel[ii] = []
        counts_perpanel[ii] = np.zeros((sdim, fdim))
        panel_keys[pid] = ii

    centroid_x, centroid_y, _ = map(lambda x: np.array(x)-0.5, prediction_utils.xyz_from_refl(exper_refls_strong))
    for i,j, pid in zip(centroid_x, centroid_y, panel_ids):
        i1 = int(max(0, i-bb))
        i2 = int(min(fdim, i+bb))
        j1 = int(max(0, j-bb))
        j2 = int(min(sdim, j+bb))
        roi = (i1, i2), (j1, j2)
        ii = panel_keys[pid]
        rois_perpanel[ii].append(roi)
        counts_perpanel[ii][j1:j2, i1:i2] += 1

    # <><><><><><><><><>
    # DO THE SIMULATION
    # <><><><><><><><><>
    # choose a device Id for GPU
    device_Id = np.random.choice(range(n_gpu))
    # call the simulation helper
    simsAB = sim_utils.sim_colors(
        crystal, DET, BEAM, FF,
        energies, fluences, pids=panels_with_spots, 
        profile=profile, cuda=not args.nocuda, oversample=1,
        Ncells_abc=Ncells_abc, mos_dom=1, mos_spread=0,
        master_scale=1, recenter=True,
        roi_pp=rois_perpanel, counts_pp=counts_perpanel,
        exposure_s=exposure_s, beamsize_mm=beamsize_mm, device_Id=device_Id,
        show_params=args.show_params, accumulate=True, crystal_size_mm=xtal_size)

    embed()

    ## make the spot integration foreground mask from the predictions
    #panel_integration_masks = {}
    #for pid in range(n_panels):
    #    panel_integration_masks[pid] = None
    #
    ## group predictions bty panel name
    #refls_predict_bypanel = prediction_utils.refls_by_panelname(refls_predict)
    #for panel_id in refls_predict_bypanel:
    #    fast, slow = DET[int(panel_id)].get_image_size()
    #    mask = prediction_utils.strong_spot_mask_dials(refls_predict_bypanel[panel_id], (slow, fast),
    #                                  as_composite=True)
    #    # if the panel mask is not set, set it!
    #    if panel_integration_masks[panel_id] is None:
    #        panel_integration_masks[panel_id] = mask
    #    # otherwise add to it   NOTE I think this is only for polychromatic...
    #    else:
    #        panel_integration_masks[panel_id] = np.logical_or(mask, panel_integration_masks[panel_id])

    ## <><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    ## HERE WE LOAD THE STRONG SPOTS AND MAKE THEM INTO A MASK that can then be combined with the integration mask
    ## such that the combination mask essentially tells us which pixels are background pixels
    ## <><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    ## load strong spot reflections, these should be named after the images by stills process, hence the string manipulation
    #refl_strong_fname = os.path.join(indexdirname,
    #            "idx-"+os.path.basename(fpath.replace(".npz", "_strong.refl")))
    #refls_strong = flex.reflection_table.from_file(refl_strong_fname) 
    ## make mask of all strong spot pixels..
    #nfast, nslow = DET[0].get_image_size()
    #img_shape = nslow, nfast  # numpy format
    ## make a mask that tells me True if I am a background pixel
    #is_bg_pixel = np.ones((n_panels, nslow, nfast), bool)
    ## group the refls by panel ID
    #refls_strong_perpan = prediction_utils.refls_by_panelname(refls_strong)
    #for panel_id in refls_strong_perpan:
    #    fast, slow = DET[int(panel_id)].get_image_size()
    #    mask = prediction_utils.strong_spot_mask_dials(
    #        refls_strong_perpan[panel_id], (slow, fast),
    #        as_composite=True)
    #    # dilate the mask
    #    mask = binary_dilation(mask, iterations=args.dilate)
    #    is_bg_pixel[panel_id] = ~mask  # strong spots should not be background pixels

    ## Combine strong spot mask and integration mask, both with dilations, to get the best
    ## possible selection of background pixels..
    #for i_predict in range(n_predict):
    #    ref_predict = refls_predict[i_predict]
    #    i1, i2, j1, j2, _, _ = ref_predict['bbox']
    #    i_panel = ref_predict['panel']
    #    integration_mask = panel_integration_masks[i_panel][j1:j2, i1:i2]
    #    # expand the integration mask so as not to include background pixels near it
    #    expanded_integration_mask = binary_dilation(integration_mask, iterations=args.dilate)

    #    # get the pixels already marked as background
    #    bg = is_bg_pixel[i_panel, j1:j2, i1:i2]
    #    # update the background pixel selection with the expanded integration mask
    #    is_bg_pixel[i_panel, j1:j2, i1:i2] = ~np.logical_or(~bg, expanded_integration_mask)

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    #   RUN THE TILT PLANE HELPER FUNCTION
    # 1. Weighted fit of the background tilt plane
    # 2. Updates prediction reflection table with miller indices, and shoeboxes
    # 3. Updates prediction reflections with integrations and integration variances
    #    using Leslie 99
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    #exper = Experiment()
    #exper.detector = DET
    #exper.beam = BEAM
    #exper.crystal = crystal
    #exper.imageset = iset  # Exper.imageset

    #if not args.oldway:
    #    refls_predict = TiltPlanes.prep_relfs_for_tiltalization(refls_predict, exper=exper)

    #    tiltnation = TiltPlanes(panel_imgs=img_data, panel_bg_masks=is_bg_pixel, panel_badpix_masks=None)
    #    tiltnation.check_if_refls_are_formatted_for_this_class(refls_predict)
    #    tiltnation.make_quick_bad_pixel_proximity_checker(refls_predict)
    #    tiltnation.sigma_rdout = sigma_readout
    #    tiltnation.adu_per_photon = GAIN
    #    tiltnation.delta_Q = args.deltaq   #0.085
    #    tiltnation.zinger_zscore = args.Z
    #    detector_node = DET[0]  # all nodes should have same pixel size, detector distance, and dimension
    #    tiltnation.pixsize_mm = detector_node.get_pixel_size()[0]
    #    tiltnation.detdist_mm = detector_node.get_distance()
    #    tiltnation.ave_wavelength_A = BEAM.get_wavelength()
    #    tiltnation.min_background_pix_for_fit = 10
    #    tiltnation.min_dist_to_bad_pix = 7

    #    all_residual = []
    #    mins = []
    #    bboxes = []
    #    tilt_abc = []
    #    error_in_tilt = []
    #    I_Leslie99 = []
    #    varI_Leslie99 = []
    #    did_i_index = []
    #    boundary_spot = []
    #    bbox_panel_ids = []
    #    Hi = []

    #    indexed_Hi = []
    #    selected_ref_idx = []
    #    all_reso = []
    #    all_fit_sel =[]
    #    all_below_zero = []
    #    for i_r in range(n_predict):
    #        ref = refls_predict[i_r]
    #        mil_idx = [int(hi) for hi in ref["miller_index"]]

    #        if mil_idx == [0, 0, 0]:
    #            continue

    #        if mil_idx in indexed_Hi:
    #            print("already indexed, this split across two panels!")
    #            continue

    #        result = tiltnation.integrate_shoebox(ref)
    #        if result is None:
    #            continue
    #        shoebox_roi, coefs, variance_matrix, Isum, varIsum, below_zero_flag, fit_sel = result
    #        if below_zero_flag and not args.keepbelowzero:
    #            print("Tilt plane dips below 0!")
    #            continue
    #        else:
    #            all_below_zero.append(below_zero_flag) 
    #        bboxes.append(shoebox_roi)
    #        tilt_abc.append(coefs)
    #        error_in_tilt.append(np.diag(variance_matrix).sum() )
    #        I_Leslie99.append(Isum)
    #        varI_Leslie99.append(varIsum)
    #        bbox_panel_ids.append(int(ref["panel"]))
    #        Hi.append(mil_idx)
    #        did_i_index.append(True)
    #        if args.savefitsel:
    #            all_fit_sel.append(fit_sel)
    #        x1, x2, y1, y2 = shoebox_roi
    #        if x1 == 0 or y1 == 0 or x2 == fs_dim or y2 == ss_dim:
    #            boundary_spot.append(True)
    #        else:
    #            boundary_spot.append(False)
    #        indexed_Hi.append(mil_idx)
    #        selected_ref_idx.append(i_r)
    #        reso = 1./np.linalg.norm(ref['rlp'])
    #        all_reso.append(reso)

    #    chosen_selection = flex.bool([i in selected_ref_idx for i in range(n_predict)])
    #    refls_predict = refls_predict.select(chosen_selection)
    #    spot_snr = np.array(I_Leslie99) / np.sqrt(varI_Leslie99)
    #    spot_snr[np.isnan(spot_snr)] = -999  # sometimes variance is 0 or < 0, leading to nan snr values..
    #   
    #    # make a padded numpy array for storing those pixels which were used to fit the background 
    #    if args.savefitsel:
    #        maxY, maxX = np.max([sel.shape  for sel in all_fit_sel], axis=0)
    #        master_fit_sel = np.zeros((len(all_fit_sel), maxY, maxX), bool)
    #        for i_sel, sel in enumerate(all_fit_sel):
    #            ydim, xdim = sel.shape
    #            master_fit_sel[i_sel, :ydim, :xdim] = sel

    #if rank == 0 and args.showcompleteness:
    #    all_bragg_hi = utils.map_hkl_list(all_bragg_hi)
    #    all_bragg_hi = list(set(all_bragg_hi))
    #    bragg_mset = miller.set(symm, flex.miller_index(all_bragg_hi), anomalous_flag=True)
    #    bragg_mset.setup_binner(d_max=999, d_min=2, n_bins=10)
    #    print("nanoBragg predictions:\n<><><><><><><><><><><><>")
    #    bragg_mset.completeness(use_binning=True).show()

    # <><><><><><><><><><><><><><><>
    # DO THE SANITY PLOTS (OPTIONAL)
    # <><><><><><><><><><><><><><><>
    #if rank == 0 and args.sanityplots:
    #    import pandas
    #    df = pandas.DataFrame({"bbox": bboxes, "panel": bbox_panel_ids})  #,"xyzobs.px.value": xyzobs})
    #    #refls_predict_bypanel = prediction_utils.refls_by_panelname(refls_predict)
    #    refls_predict_bypanel = df.groupby("panel")
    #    pause = args.pause
    #    for panel_id in df.panel.unique():
    #        panel_id = int(panel_id)
    #        panel_img = img_data[panel_id]
    #        m = panel_img.mean()
    #        s = panel_img.std()
    #        vmax = m + 4*s
    #        vmin = m - s
    #        ax.clear()
    #        im = ax.imshow(panel_img, vmax=vmax, vmin=vmin)
    #        #int_mask = np.zeros(panel_img.shape).astype(np.bool)
    #        #bg_mask = np.zeros(panel_img.shape).astype(np.bool)
    #        df_p = refls_predict_bypanel.get_group(panel_id)
    #        ax.set_title("Panel %d" % panel_id)
    #        for i_ref in range(len(df_p)):
    #            #ref = refls_predict_bypanel[panel_id][i_ref]
    #            ref = df_p.iloc[i_ref]
    #            i1, i2, j1, j2 = ref['bbox']
    #            rect = plt.Rectangle(xy=(i1, j1), width=i2-i1, height=j2-j1, fc='none', ec='Deeppink')
    #            plt.gca().add_patch(rect)
    #            #mask = ref['shoebox'].mask.as_numpy_array()[0]
    #            #int_mask[j1:j2, i1:i2] = np.logical_or(mask == 5, int_mask[j1:j2, i1:i2])
    #            #bg_mask[j1:j2, i1:i2] = np.logical_or(mask == 19, bg_mask[j1:j2, i1:i2])
    #        plt.draw()
    #        plt.pause(pause)
    #        #im.set_data(int_mask)
    #        #plt.title("panel%d: integration mask" % panel_id)
    #        #im.set_clim(0, 1)
    #        #plt.draw()
    #        #plt.pause(pause)
    #        #im.set_data(bg_mask)
    #        #plt.title("panel%d: background mask" % panel_id)
    #        #im.set_clim(0, 1)
    #        #plt.draw()
    #        #plt.pause(pause)

    #all_paths.append(fpath)
    #all_Amats.append(crystal.get_A())
    #if rank == 0:
    #    print("Rank%d: writing" % rank)
    ## save the output!
    #writer.create_dataset("bboxes/shot%d" % n_processed, data=bboxes,  dtype=np.int, compression="lzf" )
    #writer.create_dataset("tilt_abc/shot%d" % n_processed, data=tilt_abc,  dtype=np.float32, compression="lzf" )
    #writer.create_dataset("tilt_error/shot%d" % n_processed, data=error_in_tilt,  dtype=np.float32, compression="lzf" )
    #writer.create_dataset("SNR_Leslie99/shot%d" % n_processed, data=spot_snr, dtype=np.float32, compression="lzf" )
    #writer.create_dataset("resolution/shot%d" % n_processed, data=all_reso, dtype=np.float32, compression="lzf" )
    #writer.create_dataset("I_Leslie99/shot%d" % n_processed, data=I_Leslie99, dtype=np.float32, compression="lzf" )
    #writer.create_dataset("varI_Leslie99/shot%d" % n_processed, data=varI_Leslie99, dtype=np.float32, compression="lzf" )
    #writer.create_dataset("Hi/shot%d" % n_processed, data=Hi, dtype=np.int, compression="lzf")
    #writer.create_dataset("indexed_flag/shot%d" % n_processed, data=did_i_index, dtype=np.int, compression="lzf")
    #writer.create_dataset("is_on_boundary/shot%d" % n_processed, data=boundary_spot, dtype=np.bool, compression="lzf")
    #writer.create_dataset("panel_ids/shot%d" % n_processed, data=bbox_panel_ids, dtype=np.int, compression="lzf")
    #if args.keepbelowzero:
    #    writer.create_dataset("below_zero/shot%d" % n_processed, data=all_below_zero, dtype=np.bool, compression="lzf")
    #if args.savefitsel: 
    #    writer.create_dataset("fit_selection/shot%d" %  n_processed, data=master_fit_sel, dtype=np.bool, compression="lzf")
    #
    ## add the default peak selection flags (default is all True, so select all peaks for refinement)
    #keepers = np.ones(len(bboxes)).astype(np.bool)
    #writer.create_dataset("bboxes/keepers%d" % n_processed, data=keepers, dtype=np.bool, compression="lzf")

    #if rank==0:
    #    print("Rank%d: Done writing" % rank)
    #n_processed += 1

#writer.create_dataset("Amatrices", data=all_Amats, compression="lzf")
#writer.create_dataset("h5_path", data=np.array(all_paths, dtype="S"), compression="lzf")
#writer.close()
