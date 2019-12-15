#!/usr/bin/env libtbx.python

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    has_mpi = True
except ImportError:
    rank = 0
    size = 1
    has_mpi = False
from numpy import load as np_load

from dxtbx.model.detector import DetectorFactory
det_from_dict = DetectorFactory.from_dict
from dxtbx.model.beam import BeamFactory
beam_from_dict = BeamFactory.from_dict
from simtbx.diffBragg.refiners.global_refiner import FatRefiner
from cxid9114.utils import map_hkl_list
import sys
from IPython import embed

# import functions on rank 0 only
if rank == 0:
    print("Rank0 imports")
    import time
    from argparse import ArgumentParser
    parser = ArgumentParser("Load and refine bigz")
    parser.add_argument("--readoutless",action="store_true")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--Ncells_size", default=30, type=float)
    parser.add_argument("--Nmos", default=1, type=int)
    parser.add_argument("--mosspread", default=0, type=float)
    parser.add_argument("--gainval", default=28, type=float)
    parser.add_argument("--curseoftheblackpearl", action="store_true")
    parser.add_argument("--outdir", type=str, default=None, help="where to write output files")
    parser.add_argument("--noiseless", action="store_true")
    parser.add_argument("--stride", type=int, default=10, help='plot stride')
    parser.add_argument("--boop", action="store_true")
    parser.add_argument("--residual", action='store_true')
    parser.add_argument('--filterbad', action='store_true')
    parser.add_argument("--maxcalls", type=int, default=30000)
    parser.add_argument("--tryscipy", action="store_true")
    parser.add_argument("--restartfile", type=str, default=None)
    parser.add_argument("--sad", action="store_true")
    parser.add_argument("--symbol", default="P43212", type=str)
    parser.add_argument("--bg", action="store_true")
    parser.add_argument("--p9", action="store_true")
    parser.add_argument("--bs7", action="store_true")
    parser.add_argument("--bs7real", action="store_true")
    parser.add_argument("--loadonly", action="store_true")
    parser.add_argument("--poissononly", action="store_true")
    parser.add_argument("--boopi", type=int, default=0)
    parser.add_argument("--Nmax", type=int, default=-1, help='NOT USING. Max number of images to process per rank')
    parser.add_argument("--nload", type=int, default=None, help='Max number of images to load per rank')
    parser.add_argument("--perimage", action="store_true")
    parser.add_argument('--perturblist', default=None, type=int)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--forcemono", action='store_true')
    parser.add_argument("--gainrefine", action="store_true")
    parser.add_argument("--fcellbump", default=0.1, type=float)
    parser.add_argument("--oversample", default=0, type=int)
    parser.add_argument("--hack", action="store_true", help="use the local 6 tester files")
    parser.add_argument("--curvatures", action='store_true')
    parser.add_argument("--startwithtruth", action='store_true')
    parser.add_argument("--startwithopt", action="store_true")
    parser.add_argument("--testmode2", action="store_true", help="debug flag for doing a test run")
    parser.add_argument("--glob", type=str, required=True, help="glob for selecting files (output files of process_mpi")
    parser.add_argument("--partition", action="store_true")
    parser.add_argument("--partitiontime", default=5, type=float, help="seconds allowed for partitioning inputs")
    parser.add_argument("--keeperstag", type=str, default="keepers", help="name of keepers boolean array")
    parser.add_argument("--plotstats", action="store_true")
    parser.add_argument("--umatrix", action="store_true")
    parser.add_argument("--bmatrix", action="store_true")
    parser.add_argument("--fcell", action="store_true")
    parser.add_argument("--ncells", action="store_true")
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--plotfcell", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--perturbfcell", default=None, type=float)

    args = parser.parse_args()
    import os
    if args.outdir is not None:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    from h5py import File as h5py_File
    from cxid9114.integrate.integrate_utils import Integrator
    from numpy import array, sqrt, percentile
    from numpy import indices as np_indices
    from numpy import zeros as np_zeros
    from numpy import sum as np_sum
    from psutil import Process
    from glob import glob
    from os import getpid
    from numpy import load as numpy_load
    from resource import getrusage
    from cxid9114.helpers import compare_with_ground_truth
    import resource
    RUSAGE_SELF = resource.RUSAGE_SELF
    from simtbx.diffBragg.refiners.crystal_systems import TetragonalManager
    from dxtbx.model import Crystal
    from scitbx.matrix import sqr
    from simtbx.diffBragg.sim_data import SimData
    from simtbx.diffBragg.nanoBragg_beam import nanoBragg_beam
    from simtbx.diffBragg.nanoBragg_crystal import nanoBragg_crystal
    from simtbx.diffBragg.refiners import RefineAllMultiPanel
    #from cxid9114.geom.multi_panel import CSPAD_refined as CSPAD
    from cxid9114.geom.multi_panel import CSPAD
    from cctbx.array_family import flex
    from cctbx import sgtbx, miller

    # let the root load the structure factors and energies to later broadcast
    from cxid9114.sf import struct_fact_special
    sf_path = os.path.dirname(struct_fact_special.__file__)
    sfall_file = os.path.join(sf_path, "realspec_sfall.h5")
    data_sf, data_energies = struct_fact_special.load_sfall(sfall_file)
    from cxid9114.parameters import ENERGY_CONV, ENERGY_LOW
    import numpy as np
    # grab the structure factors at the edge energy (ENERGY_LOW=8944 eV)
    edge_idx = np.abs(data_energies -ENERGY_LOW).argmin()
    Fhkl_guess = data_sf[edge_idx]
    wavelens = ENERGY_CONV / data_energies

    from dials.algorithms.indexing.compare_orientation_matrices import difference_rotation_matrix_axis_angle as diff_rot

else:
    np_indices = None
    sf_path = None
    diff_rot = None
    compare_with_ground_truth = None
    args = None
    RefineAllMultiPanel = None
    nanoBragg_beam = nanoBragg_crystal = None
    SimData = None
    # beam_from_dict = det_from_dict = None
    h5py_File = None
    Integrator = None
    CSPAD = None
    array = sqrt = percentile = np_zeros = np_sum = None
    Process = None
    glob = None
    getpid = None
    numpy_load = None
    RUSAGE_SELF = None
    getrusage = None
    TetragonalManager = None
    Crystal = None
    sqr = None
    Fhkl_guess = wavelens = None


if has_mpi:
    if rank == 0:
        print("Broadcasting imports")
    #FatRefiner = comm.bcast(FatRefiner, root=0)
    RefineAllMultiPanel = comm.bcast(RefineAllMultiPanel)
    np_indices = comm.bcast(np_indices, root=0)
    glob = comm.bcast(glob, root=0)
    diff_rot = comm.bcast(diff_rot, root=0)
    compare_with_ground_truth = comm.bcast(compare_with_ground_truth, root=0)
    args = comm.bcast(args, root=0)
    sf_path = comm.bcast(sf_path, root=0)
    Fhkl_guess = comm.bcast(Fhkl_guess, root=0)
    wavelens = comm.bcast(wavelens, root=0)
    Crystal = comm.bcast(Crystal, root=0)
    sqr = comm.bcast(sqr, root=0)
    CSPAD = comm.bcast(CSPAD, root=0)
    nanoBragg_beam = comm.bcast(nanoBragg_beam, root=0)
    nanoBragg_crystal = comm.bcast(nanoBragg_crystal, root=0)
    SimData = comm.bcast(SimData, root=0)
    # beam_from_dict = comm.bcast(beam_from_dict, root=0)
    # det_from_dict = comm.bcast(det_from_dict, root=0)
    h5py_File = comm.bcast(h5py_File, root=0)
    Integrator = comm.bcast(Integrator, root=0)
    array = comm.bcast(array, root=0)
    sqrt = comm.bcast(sqrt, root=0)
    percentile = comm.bcast(percentile, root=0)
    np_zeros = comm.bcast(np_zeros, root=0)
    np_sum = comm.bcast(np_sum, root=0)
    Process = comm.bcast(Process, root=0)
    getpid = comm.bcast(getpid, root=0)
    numpy_load = comm.bcast(numpy_load, root=0)
    getrusage = comm.bcast(getrusage, root=0)
    RUSAGE_SELF = comm.bcast(RUSAGE_SELF, root=0)
    TetragonalManager = comm.bcast(TetragonalManager, root=0)


class FatData:

    def __init__(self):
        self.int_radius = 5  #
        self.gain = args.gainval  # gain of panels, can be refined, can be panel dependent
        self.symbol = args.symbol
        self.anomalous_flag = True
        self.flux_min = 1e2  # minimum number of photons to simulate (assume flux is N-photons, e.g. 1 second exposure)
        self.n_ucell_param = 2  # tetragonal cell
        self.Nload = args.nload  #
        self.all_pix = 0
        self.time_load_start = 0
        self.fnames = []  # the filenames containing the datas
        self.per_image_refine_first = args.perimage  # do a per image refinement of crystal model prior to doing the global fat
        self.all_spot_roi = {}  # one list per shot, rois are x1,x2,y1,y2 per reflection
        self.all_abc_inits = {}  # one list per shot, abc_inits are a,b,c per reflection
        self.all_panel_ids = {}  # one list per shot, panel_ids are single number per reflection
        self.all_ucell_mans = {}  # one per shot, UcellManager instance (Tetragonal in this case)
        self.all_spectra = {}  # one list of (wavelength, flux) tuples per shot
        self.all_crystal_models = {}
        self.all_crystal_GT = {}
        self.all_xrel = {}
        self.all_yrel = {}
        self.all_Hi_asu = {}
        self.all_crystal_scales = {}
        self.all_Hi = {}
        self.all_nanoBragg_rois = {}
        self.SIM = None  # simulator; one per rank!
        self.all_roi_imgs = {}
        self.all_fnames = {}
        self.all_proc_idx = {}
        self.all_proc_fnames = {}
        self.nbbeam = self.nbcryst = None
        self.miller_data_map = None

    def initialize_simulator(self, init_crystal, init_beam, init_spectrum, init_miller_array):
        # create the sim_data instance that the refiner will use to run diffBragg
        # create a nanoBragg crystal
        self.nbcryst = nanoBragg_crystal()
        self.nbcryst.dxtbx_crystal = init_crystal
        self.nbcryst.thick_mm = 0.1
        self.nbcryst.Ncells_abc = args.Ncells_size, args.Ncells_size, args.Ncells_size

        self.nbcryst.miller_array = init_miller_array
        self.nbcryst.n_mos_domains = args.Nmos
        self.nbcryst.mos_spread_deg = args.mosspread

        # create a nanoBragg beam
        self.nbbeam = nanoBragg_beam()
        self.nbbeam.size_mm = 0.001
        self.nbbeam.unit_s0 = init_beam.get_unit_s0()
        self.nbbeam.spectrum = init_spectrum

        # sim data instance
        self.SIM = SimData()
        self.SIM.detector = CSPAD
        self.SIM.crystal = self.nbcryst
        self.SIM.beam = self.nbbeam
        self.SIM.panel_id = 0  # default
        self.SIM.instantiate_diffBragg(default_F=0, oversample=args.oversample)
        if args.sad:
            if args.p9:
                self.SIM.D.spot_scale = 3050
            elif args.bs7 or args.bs7real:
                self.SIM.D.spot_scale = 250
            else:
                self.SIM.D.spot_scale = .7
        else:
            self.SIM.D.spot_scale = 12

    def _process_miller_data(self):
        idx, data = self.SIM.D.Fhkl_tuple
        self.miller_data_map = {idx: val for idx, val in zip(idx, data)}

    # @profile
    def load(self):

        # some parameters

        # NOTE: for reference, inside each h5 file there is
        #   [u'Amatrices', u'Hi', u'bboxes', u'h5_path']

        # get the total number of shots using worker 0
        if rank == 0:
            self.time_load_start = time.time()
            print("I am root. I am calculating total number of shots")
            h5s = [h5py_File(f, "r") for f in self.fnames]
            Nshots_per_file = [h["h5_path"].shape[0] for h in h5s]
            Nshots_tot = sum(Nshots_per_file)
            print("I am root. Total number of shots is %d" % Nshots_tot)

            print("I am root. I will divide shots amongst workers.")
            shot_tuples = []
            roi_per = []
            for i_f, fname in enumerate(self.fnames):
                fidx_shotidx = [(i_f, i_shot) for i_shot in range(Nshots_per_file[i_f])]
                shot_tuples += fidx_shotidx

                # store the number of usable roi per shot in order to divide shots amongst ranks equally
                roi_per += [sum(h5s[i_f]["bboxes"]["%s%d" % (args.keeperstag,i_shot)][()])
                            for i_shot in range(Nshots_per_file[i_f])]

            from numpy import array_split
            from numpy.random import permutation
            print ("I am root. Number of uniques = %d" % len(set(shot_tuples)))

            # divide the array into chunks of roughly equal sum (total number of ROI)
            if args.partition and args.restartfile is None:
                diff = np.inf
                roi_per = np.array(roi_per)
                tstart = time.time()
                best_order = range(len(roi_per))
                print("Partitioning for better load balancing across ranks.. ")
                while 1:
                    order = permutation(len(roi_per))
                    res = [sum(a) for a in np.array_split(roi_per[order], size)]
                    new_diff = max(res) - min(res)
                    t_elapsed = time.time() - tstart
                    t_remain = args.partitiontime - t_elapsed
                    if new_diff < diff:
                        diff = new_diff
                        best_order = order.copy()
                        print("Best diff=%d, Parition time remaining: %.3f seconds" % (diff, t_remain))
                    if t_elapsed > args.partitiontime:
                        break
                shot_tuples = [shot_tuples[i] for i in best_order]

            elif args.partition and args.restartfile is not None:
                print ("Warning: skipping partitioning time to use shot mapping as laid out in restart file dir")
            else:
                print ("Proceeding without partitioning")

            shots_for_rank = array_split(shot_tuples, size)
            import os # FIXME, I thought I was imported already!
            if args.outdir is not None:  # save for a fast restart (shot order is important!)
                np.save(os.path.join(args.outdir, "shots_for_rank"), shots_for_rank)
            if args.restartfile is not None:
                # the directory containing the restart file should have a shots for rank file
                dirname = os.path.dirname(args.restartfile)
                print ("Loading shot mapping from dir %s" % dirname)
                shots_for_rank = np.load(os.path.join(dirname, "shots_for_rank.npy"))
                # propagate the shots for rank file...
                np.save(os.path.join(args.outdir, "shots_for_rank"), shots_for_rank)
            # close the open h5s..
            for h in h5s:
                h.close()

        else:
            Nshots_tot = None
            shots_for_rank = None
            h5s = None

        # Nshots_tot = comm.bcast( Nshots_tot, root=0)
        if has_mpi:
            shots_for_rank = comm.bcast(shots_for_rank, root=0)
        # h5s = comm.bcast( h5s, root=0)  # pull in the open hdf5 files

        my_shots = shots_for_rank[rank]
        if self.Nload is not None:
            my_shots = my_shots[:self.Nload]

        # open the unique filenames for this rank
        # TODO: check max allowed pointers to open hdf5 file
        my_unique_fids = set([fidx for fidx, _ in my_shots])
        my_open_files = {fidx: h5py_File(self.fnames[fidx], "r") for fidx in my_unique_fids}
        Ntot = 0

        for img_num, (fname_idx, shot_idx) in enumerate(my_shots):
            #if img_num == args.Nmax:
            #    # print("Already processed maximum number images!")
            #    continue
            h = my_open_files[fname_idx]

            # load the dxtbx image data directly:
            npz_path = h["h5_path"][shot_idx]
            if args.testmode2:
                import os
                npz_path = npz_path.split("d9114_sims/")[1]
                npz_path = os.path.join("/Users/dermen/", npz_path)
            if args.noiseless:
                noiseless_path = npz_path.replace(".npz", ".noiseless.npz")
                img_handle = numpy_load(noiseless_path)
            elif args.readoutless:
                import os
                #readoutless_path = npz_path.split("tang/")[1]
                #readoutless_path = os.path.join("/global/project/projectdirs/lcls/dermen/d9114_sims/bear",
                #                                readoutless_path)
                readoutless_path = npz_path.replace("tang", "bear")
                img_handle = numpy_load(readoutless_path)
            else:
                img_handle = numpy_load(npz_path)

            img = img_handle["img"]

            if len(img.shape) == 2:  # if single panel
                img = array([img])

            # D = det_from_dict(img_handle["det"][()])
            B = beam_from_dict(img_handle["beam"][()])

            # get the indexed crystal Amatrix
            Amat = h["Amatrices"][shot_idx]
            amat_elems = list(sqr(Amat).inverse().elems)
            # real space basis vectors:
            a_real = amat_elems[:3]
            b_real = amat_elems[3:6]
            c_real = amat_elems[6:]

            # dxtbx indexed crystal model
            C = Crystal(a_real, b_real, c_real, "P43212")

            ## NOTE: this is a temporary hack
            if args.startwithopt:
                exit()
                assert args.hack
                f_basename = os.path.basename(npz_path)
                crystal_dir = "/Users/dermen/crystal/modules/cxid9114/io/crystals/"
                crystal_name = crystal_dir + "/%s_refined.npy" % f_basename
                assert os.path.exists(crystal_name), "Crystal file non existent"
                refined_Amat = np_load(crystal_name)
                C.set_A(tuple(refined_Amat))
            ## NOTE end hack

            # change basis here ? Or maybe just average a/b
            a, b, c, _, _, _ = C.get_unit_cell().parameters()
            a_init = .5 * (a + b)
            c_init = c

            # shoe boxes where we expect spots
            bbox_dset = h["bboxes"]["shot%d" % shot_idx]
            n_bboxes_total = bbox_dset.shape[0]
            # is the shoe box within the resolution ring and does it have significant SNR (see filter_bboxes.py)
            is_a_keeper = h["bboxes"]["%s%d" % (args.keeperstag, shot_idx)][()]

            # tilt plane to the background pixels in the shoe boxes
            tilt_abc_dset = h["tilt_abc"]["shot%d" % shot_idx]
            # miller indices (not yet reduced by symm equivs)
            Hi_dset = h["Hi"]["shot%d" % shot_idx]
            try:
                panel_ids_dset = h["panel_ids"]["shot%d" % shot_idx]
                has_panels = True
            except KeyError:
                has_panels = False

            # apply the filters:
            bboxes = [bbox_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            tilt_abc = [tilt_abc_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            Hi = [tuple(Hi_dset[i_bb]) for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            proc_file_idx = [i_bb for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]

            if has_panels:
                panel_ids = [panel_ids_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            else:
                panel_ids = [0] * len(tilt_abc)

            # how many pixels do we have
            tot_pix = [(j2 - j1)*(i2 - i1) for i1, i2, j1, j2 in bboxes]
            Ntot += sum(tot_pix)

            # load some ground truth data from the simulation dumps (e.g. spectrum)
            h5_fname = h["h5_path"][shot_idx].replace(".npz", "")
            if args.testmode2:
                h5_fname = npz_path.split(".npz")[0]
            data = h5py_File(h5_fname, "r")

            xtal_scale_truth = data["spot_scale"][()]
            tru = sqr(data["crystalA"][()]).inverse().elems
            a_tru = tru[:3]
            b_tru = tru[3:6]
            c_tru = tru[6:]
            C_tru = Crystal(a_tru, b_tru, c_tru, "P43212")
            try:
                angular_offset_init = compare_with_ground_truth(a_tru, b_tru, c_tru, [C], symbol="P43212")[0]
            except Exception as err:
                print("Rank %d: Boo cant use the comparison w GT function: %s" % (rank, err))

            fluxes = data["spectrum"][()]
            es = data["exposure_s"][()]

            #spec = fluxes * es
            #spec_f = h5py_File("/Users/dermen/crystal/modules/cxid9114/spec/realspec.h5", "r")
            #vals = spec_f["hist_spec"][()]
            #res = np.array([np.allclose(spec, v) for v in vals])
            #idx = np.where(res)[0]
            #if not idx.size==1:
            #    print ("Rank %d couldnt get it" % rank)
            #else:
            #    idx = idx[0]
            #    run = spec_f["runs"][idx]
            #    _shot = spec_f["shot_idx"][idx]
            #    print "Rank %d: filename %s, run %f, shot %d" % (rank, npz_path, run, _shot)

            #comm.Barrier()

            #exit()
            fluxes *= es  # multiply by the exposure time
            # TODO: wavelens should come from the imageset file itself
            wavelens = data["wavelengths"] [()]
            spectrum = zip(wavelens, fluxes)
            # dont simulate when there are no photons!
            spectrum = [(wave, flux) for wave, flux in spectrum if flux > self.flux_min]

            if args.forcemono:
                spectrum = [(B.get_wavelength(), sum(fluxes))]

            # make a unit cell manager that the refiner will use to track the B-matrix
            aa, _, cc, _, _, _ = C_tru.get_unit_cell().parameters()
            ucell_man = TetragonalManager(a=a_init, c=c_init)
            if args.startwithtruth:
                ucell_man = TetragonalManager(a=aa, c=cc)

            if args.startwithtruth:
                C = C_tru
            # create the sim_data instance that the refiner will use to run diffBragg
            # create a nanoBragg crystal
            if img_num == 0:  # only initialize the simulator after loading the first image
                if args.sad:
                    if args.p9:
                        wavelen = 0.9793
                        from cxid9114.sf.struct_fact_special import load_p9
                        Fhkl_guess = load_p9()
                    elif args.bs7 or args.bs7real:
                        from cxid9114.parameters import WAVELEN_HIGH
                        from cxid9114.sf import struct_fact_special
                        import os
                        wavelen = WAVELEN_HIGH
                        Fhkl_guess = struct_fact_special.sfgen(WAVELEN_HIGH, 
                            os.path.join(sf_path, "../sim/4bs7.pdb"), 
                            yb_scatter_name=os.path.join(sf_path, "../sf/scanned_fp_fdp.npz"))
                    else:
                        from cxid9114.parameters import WAVELEN_LOW
                        wavelen = WAVELEN_LOW
                        from cxid9114.sf.struct_fact_special import load_4bs7_sf
                        Fhkl_guess = load_4bs7_sf()

                    if not args.bs7real:
                        spectrum = [(wavelen, fluxes[0])]
                    # end if sad
                self.initialize_simulator(C, B, spectrum, Fhkl_guess.as_amplitude_array())

            # map the miller array to ASU
            Hi_asu = map_hkl_list(Hi, self.anomalous_flag, self.symbol)
            #sg_type = sgtbx.space_group_info(symbol=self.symbol).type()
            #Hi_flex = flex.miller_index(tuple(map(tuple, Hi)))
            #miller.map_to_asu(sg_type, self.anomalous_flag, Hi_flex)  # mods Hi_flex in place
            #Hi_asu = list(Hi_flex)

            # copy the image as photons (NOTE: Dont forget to ditch its references!)
            img_in_photons = (img/args.gainval).astype('float32')

            # Here, takeout from the image only whats necessary to perform refinement
            # first filter the spot rois so they dont occur exactly at the boundary of the image (inclusive range in nB)
            assert len(img_in_photons.shape) == 3  # sanity
            nslow, nfast = img_in_photons[0].shape
            bboxes = array(bboxes)
            for i_bbox, (_, x2, _, y2) in enumerate(bboxes):
                if x2 == nfast:
                    bboxes[i_bbox][1] = x2 - 1  # update roi_xmax
                if y2 == nslow:
                    bboxes[i_bbox][3] = y2 - 1  # update roi_ymax
            # now cache the roi in nanoBragg format ((x1,x2), (y1,y1))
            # and also cache the pixels and the coordinates

            nanoBragg_rois = []  # special nanoBragg format
            xrel, yrel, roi_img = [], [], []
            for i_roi, (x1, x2, y1, y2) in enumerate(bboxes):
                nanoBragg_rois.append(((x1, x2), (y1, y2)))
                yr, xr = np_indices((y2 - y1 + 1, x2 - x1 + 1))
                xrel.append(xr)
                yrel.append(yr)
                pid = panel_ids[i_roi]
                roi_img.append(img_in_photons[pid, y1:y2 + 1, x1:x2 + 1])

            # make sure to clear that damn memory
            img = None
            img_in_photons = None
            del img  # not sure if needed here..
            del img_in_photons

            # peak at the memory usage of this rank
            #mem = getrusage(RUSAGE_SELF).ru_maxrss  # peak mem usage in KB
            #mem = mem / 1e6  # convert to GB
            mem = self._usage()

            print "RANK %d: %.2g total pixels in %d/%d bboxes (file %d / %d); MemUsg=%2.2g GB" \
                  % (rank, Ntot, len(bboxes), n_bboxes_total,  img_num +1, len(my_shots), mem)
            self.all_pix += Ntot

            # TODO: accumulate per-shot information
            self.all_spot_roi[img_num] = bboxes
            self.all_abc_inits[img_num] = tilt_abc
            self.all_panel_ids[img_num] = panel_ids
            self.all_ucell_mans[img_num] = ucell_man
            self.all_spectra[img_num] = spectrum
            self.all_crystal_models[img_num] = C
            self.all_crystal_scales[img_num] = xtal_scale_truth
            self.all_crystal_GT[img_num] = C_tru
            self.all_xrel[img_num] = xrel
            self.all_yrel[img_num] = yrel
            self.all_nanoBragg_rois[img_num] = nanoBragg_rois
            self.all_roi_imgs[img_num] = roi_img
            self.all_fnames[img_num] = npz_path
            self.all_proc_fnames[img_num] = h.filename
            self.all_Hi[img_num] = Hi
            self.all_Hi_asu[img_num] = Hi_asu
            self.all_proc_idx[img_num] = proc_file_idx

        for h in my_open_files.values():
            h.close()

        print ("Rank %d; all subimages loaded!" % rank)

    def _usage(self):
        mem = getrusage(RUSAGE_SELF).ru_maxrss  # peak mem usage in KB
        conv = 1e-6
        try:
            if "darwin" in sys.platform:
                conv = 1e-9
        except:
            pass
        mem = mem * conv  # convert to GB
        return mem

    def tally_statistics(self):

        # tally up all miller indices in this refinement
        self._gather_Hi_information()
        self.num_hkl_global = len(self.idx_from_asu)

        n_images = len(self.all_spot_roi)
        self.n_images = n_images
        n_spot_per_image = [len(self.all_spot_roi[i_image]) for i_image in range(n_images)]
        n_spot_tot = sum(n_spot_per_image)
        total_pix = self.all_pix
        n_rot_param = 3
        n_ncell_param = 1
        n_scale_param = 1
        # TODO background refine
        # NOTE: n_param_per_image is no longer a constant when we refine background planes (unless we do a per-image polynomial fit background plane model)
        n_param_per_image = [n_rot_param + n_ncell_param + n_scale_param + 3*n_spot_per_image[i]
                             for i in range(n_images)]
        #n_param_per_image = n_rot_param + n_ncell_param + n_scale_param
        self.n_param_per_image = n_param_per_image

        # TODO background refine
        total_per_image_unknowns = sum(n_param_per_image)  # NOTE background refine
        #total_per_image_unknowns = n_param_per_image * n_images

        self.n_local_unknowns = total_per_image_unknowns

        mem = self._usage()

        print("RANK%d: images=%d, spots=%d, pixels=%d, unknowns=%d, usage=%2.2g GigBy"
              % (comm.rank, n_images, n_spot_tot, total_pix, total_per_image_unknowns, mem))

        n_images = comm.reduce(n_images, MPI.SUM, root=0)
        n_spot_tot = comm.reduce(n_spot_tot, MPI.SUM, root=0)
        total_pix = comm.reduce(total_pix, MPI.SUM, root=0)
        # Gather so that each rank knows how many local unknowns on each rank
        local_unknowns_per_rank = comm.gather(total_per_image_unknowns, root=0)
        mem_tot = comm.reduce(mem, MPI.SUM, root=0)

        if comm.rank == 0:
            total_local_unknowns = sum(local_unknowns_per_rank)
        else:
            total_local_unknowns = None

        self.local_unknowns_across_all_ranks = comm.bcast(total_local_unknowns, root=0)
        self.n_global_params = 2 + self.n_ucell_param + self.num_hkl_global  # detdist and gain + ucell params
        self.n_total_unknowns = self.local_unknowns_across_all_ranks + self.n_global_params  # gain and detdist (originZ)

        comm.Barrier()
        if comm.rank == 0:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("MPIWORLD TOTALZ: images=%d, spots=%d, pixels=%2.2g, Nlocal/Nglboal=%d/%d, usage=%2.2g GigaBytes"
                  % (n_images, n_spot_tot, total_pix, total_local_unknowns,self.n_global_params, mem_tot))
            print("Total time elapsed= %.4f seconds" % (time.time()-self.time_load_start))
            print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")

            # determine where in the global parameter array does this rank
            # parameters begin
            starts_per_rank = {}
            xpos = 0
            for _rank, n_unknown in enumerate(local_unknowns_per_rank):
                starts_per_rank[_rank] = xpos
                xpos += n_unknown
        else:
            starts_per_rank = None

        self.starts_per_rank = comm.bcast(starts_per_rank, root=0)

    def _gather_Hi_information(self):
        nshots_on_this_rank = len(self.all_Hi)
        H, Hasu = [], []
        for i in range(nshots_on_this_rank):
            H += B.all_Hi[i]
            Hasu += B.all_Hi_asu[i]

        print("Rank %d: Num miller vars on rank=%d" % (comm.rank, len(set(Hasu))))

        Hi_all_ranks = comm.reduce(H, root=0)  # adding python lists concatenates them
        self.Hi_all_ranks = comm.bcast(Hi_all_ranks, root=0)

        Hi_asu_all_ranks = comm.reduce(Hasu, root=0)
        self.Hi_asu_all_ranks = comm.bcast(Hi_asu_all_ranks, root=0)

        # after gather
        if comm.rank == 0:
            print("Rank %d: total miller vars=%d" % (comm.rank, len(set(Hi_asu_all_ranks))))

        # this will map the measured miller indices to their index in the LBFGS parameter array self.x
        self.idx_from_asu = {h: i for i, h in enumerate(set(self.Hi_asu_all_ranks))}
        # we will need the inverse map during refinement to update the miller array in diffBragg, so we cache it here
        self.asu_from_idx = {i: h for i, h in enumerate(set(self.Hi_asu_all_ranks))}

    def refine(self):
        init_gain = 1
        if args.gainrefine:
            init_gain = 1.2
        self.RUC = FatRefiner(
            n_total_params=self.n_total_unknowns,
            n_local_params=self.n_local_unknowns,
            n_global_params=self.n_global_params,
            local_idx_start=self.starts_per_rank[comm.rank],
            shot_ucell_managers=self.all_ucell_mans,
            shot_rois=self.all_spot_roi,
            shot_nanoBragg_rois=self.all_nanoBragg_rois,
            shot_roi_imgs=self.all_roi_imgs, shot_spectra=self.all_spectra,
            shot_crystal_GTs=self.all_crystal_GT, shot_crystal_models=self.all_crystal_models,
            shot_xrel=self.all_xrel, shot_yrel=self.all_yrel, shot_abc_inits=self.all_abc_inits,
            shot_asu=self.all_Hi_asu,
            global_param_idx_start=self.local_unknowns_across_all_ranks,
            shot_panel_ids=self.all_panel_ids,
            all_crystal_scales=self.all_crystal_scales,
            init_gain=init_gain,
            perturb_fcell=args.perturbfcell)

        # plot things
        self.RUC.debug = args.debug
        self.RUC.binner_dmax = 999
        self.RUC.binner_dmin = 2
        self.RUC.binner_nbin = 10

        self.RUC.plot_images = args.plot
        self.RUC.plot_fcell = args.plotfcell
        self.RUC.plot_residuals = args.residual
        self.RUC.plot_statistics = args.plotstats
        self.RUC.setup_plots()

        self.RUC.log_fcells =True

        if args.perturblist is not None:
            self.RUC._hacked_fcells = range(args.perturblist)
        self.RUC.idx_from_asu = self.idx_from_asu
        self.RUC.asu_from_idx = self.asu_from_idx
        self.RUC.request_diag_once = False
        self.RUC.S = self.SIM
        self.RUC.restart_file = args.restartfile
        self.RUC.has_pre_cached_roi_data = True
        self.RUC.split_evaluation = False
        self.RUC.trad_conv = True
        self.RUC.fcell_bump = args.fcellbump
        self.RUC.refine_detdist = False
        self.RUC.refine_background_planes = False
        self.RUC.S.D.update_oversample_during_refinement = False
        self.RUC.refine_Umatrix = args.umatrix
        self.RUC.refine_Fcell = args.fcell
        self.RUC.refine_Bmatrix = args.bmatrix
        self.RUC.refine_ncells = args.ncells
        self.RUC.refine_crystal_scale = args.scale
        self.RUC.refine_background_planes = args.bg
        self.RUC.refine_gain_fac = args.gainrefine
        self.RUC.use_curvatures = False  # args.curvatures
        self.RUC.calc_curvatures = args.curvatures
        self.RUC.poisson_only = args.poissononly
        self.RUC.plot_stride = args.stride
        self.RUC.trad_conv_eps = 5e-10  # NOTE this is for single panel model
        self.RUC.max_calls = args.maxcalls
        self.RUC.verbose = False
        self.RUC.use_rot_priors = False
        self.RUC.use_ucell_priors = False
        self.RUC.filter_bad_shots = args.filterbad
        self.RUC.FNAMES = self.all_fnames
        self.RUC.PROC_FNAMES = self.all_proc_fnames
        self.RUC.PROC_IDX = self.all_proc_idx
        self.RUC.Hi = self.all_Hi
        self.RUC.output_dir = args.outdir

        if args.verbose:
            if rank == 0:  # only show refinement stats for rank 0
                self.RUC.verbose = True

        if args.tryscipy:
            self.RUC.calc_curvatures = False
            self.RUC._setup()
            self.RUC.calc_func = True
            self.RUC.compute_functional_and_gradients()

            from scitbx.array_family import flex
            def func(x, RUC):
                RUC.calc_func = True
                RUC.x = flex.double(x)
                f, g = RUC.compute_functional_and_gradients()
                return f

            def fprime(x, RUC):
                RUC.calc_func = False
                RUC.x = flex.double(x)
                RUC.x = flex.double(x)
                f, g = RUC.compute_functional_and_gradients()
                return g.as_numpy_array()

            from scipy.optimize import fmin_l_bfgs_b
            out = fmin_l_bfgs_b(func=func, x0=array(self.RUC.x),
                                fprime=fprime, args=[self.RUC])

            #import climin
            #opt = climin.Lbfgs(wrt=self.RUC.x.as_numpy_array(),
            #   f=func, fprime=fprime, args=self.RUC)
            #for info in opt:
            #    print ()

        else:
            self.RUC.run()
            if self.RUC.hit_break_to_use_curvatures:
                self.RUC.num_positive_curvatures = 0
                self.RUC.use_curvatures = True
                self.RUC.run(setup=False)


    def print_results(self):
        for i_shot in range(self.RUC.n_shots): # (angx, angy, angz, a,c) in enumerate(zip(rotx, roty, rotz, avals, cvals)):

            a_init, _, c_init, _, _, _ = self.RUC.CRYSTAL_MODELS[i_shot].get_unit_cell().parameters()
            a_tru, b_tru, c_tru = self.RUC.CRYSTAL_GT[i_shot].get_real_space_vectors()
            try:
                angular_offset_init = compare_with_ground_truth(a_tru, b_tru, c_tru,
                                                [self.RUC.CRYSTAL_MODELS[i_shot]],
                                                symbol="P43212")[0]
            except Exception as err:
                print("Rank %d img %d err %s" % (rank, i_shot, err))
                continue

            ang, ax = self.RUC.get_correction_misset(as_axis_angle_deg=True, i_shot=i_shot) #anglesXYZ=(angx,angy,angz))
            B = self.RUC.get_refined_Bmatrix(i_shot)
            self.RUC.CRYSTAL_MODELS[i_shot].rotate_around_origin(ax, ang)
            self.RUC.CRYSTAL_MODELS[i_shot].set_B(B)
            a_ref, _, c_ref, _, _, _ = self.RUC.CRYSTAL_MODELS[i_shot].get_unit_cell().parameters()
            # compute missorientation with ground truth model
            tot_negs = sum([(roi < 0).sum() for roi in self.RUC.ROI_IMGS[i_shot]])
            try:
                angular_offset_ref = compare_with_ground_truth(a_tru, b_tru, c_tru,
                                                               [self.RUC.CRYSTAL_MODELS[i_shot]],
                                                               symbol="P43212")[0]
                print("Rank %d, file=%s, ang=%f, init_ang=%f, a=%f, init_a=%f, c=%f, init_c=%f, total_neg_pix=%d" % (
                    rank, self.all_fnames[i_shot], angular_offset_ref, angular_offset_init, a_ref, a_init, c_ref, c_init, tot_negs))
            except Exception as err:
                print("Rank %d, file=%d, error %s" % (rank, i_shot, err))

        # free the memory from diffBragg instance

        #except AssertionError as err:
        #    print("Rank %d, Hit assertion error during refinement: %s" % (comm.rank, err))
        #    pass


B = FatData()
fnames = glob(args.glob)
B.fnames = fnames
B.load()
comm.Barrier()
B.tally_statistics()

if args.loadonly:
    comm.Abort()

B.refine()
comm.Barrier()
B.print_results()

if comm.rank == 0:
    from IPython import embed
    embed()
