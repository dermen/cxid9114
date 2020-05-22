#!/usr/bin/env libtbx.python
import cProfile

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
from simtbx.diffBragg.refiners.global_refiner import GlobalRefiner
from cxid9114.utils import open_flex
from simtbx.diffBragg.utils import map_hkl_list
import sys
from IPython import embed

# import functions on rank 0 only
if rank == 0:
    print("Rank0 imports")
    import time
    from argparse import ArgumentParser
    parser = ArgumentParser("Load and refine bigz")
    parser.add_argument("--readoutless", action="store_true")
    parser.add_argument("--checkbackground", action="store_true")
    parser.add_argument("--checkbackgroundsavename",default="_fat_data_background_residual_file", type=str, help="name of the residual background image")
    parser.add_argument("--protocol", choices=["per_shot", "global"], default="per_shot", type=str, help="refinement protocol")
    parser.add_argument("--tradeps", default=5e-10, type=float, help="traditional convergence epsilon. Convergence happens if |G| < |X|*tradeps where |G| is norm gradient and |X| is norm parameters")
    parser.add_argument("--imagecorr", action="store_true")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--fixrotZ", action='store_true')
    parser.add_argument("--Ncells_size", default=30, type=float)
    parser.add_argument("--cella", default=None, type=float)
    parser.add_argument("--gradientonly", action='store_true') 
    parser.add_argument("--cellc", default=None, type=float)
    parser.add_argument("--Nmos", default=1, type=int)
    parser.add_argument("--scipyfactr", default=1e7, type=float, help="Factor for terminating scipy lbfgs see \n https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html")
    parser.add_argument("--mosspread", default=0, type=float)
    parser.add_argument("--preopttag", default="preopt", type=str)
    parser.add_argument("--gainval", default=28, type=float)
    parser.add_argument("--curseoftheblackpearl", action="store_true", help="This argument does nothing... ")
    parser.add_argument("--ignorelinelow", action="store_true", help="ignore line search in LBFGS")
    parser.add_argument("--xrefinedonly", action="store_true" )
    parser.add_argument("--outdir", type=str, default=None, help="where to write output files")
    parser.add_argument("--imgdirname", type=str, default=None)
    parser.add_argument("--rotscale", default=1, type=float)
    parser.add_argument("--noiseless", action="store_true")
    parser.add_argument("--forcecurva", action="store_true")
    parser.add_argument("--optoutname", type=str, default="results")
    parser.add_argument("--stride", type=int, default=10, help='plot stride')
    parser.add_argument("--minmulti", type=int, default=2, help='minimum multiplicity for refinement')
    parser.add_argument("--boop", action="store_true")
    parser.add_argument("--bigdump", action="store_true")
    parser.add_argument("--residual", action='store_true')
    parser.add_argument("--NoRescaleFcellRes", action='store_true')
    parser.add_argument("--setuponly", action='store_true')
    parser.add_argument('--filterbad', action='store_true')
    parser.add_argument("--alist", type=str, default=None)
    parser.add_argument("--tryscipy", action="store_true", help="use scipy's LBFGS implementation instead of cctbx's")
    parser.add_argument("--restartfile", type=str, default=None)
    parser.add_argument("--Fobslabel", type=str, default=None)
    parser.add_argument("--Freflabel", type=str, default=None)
    parser.add_argument("--xinitfile", type=str, default=None)
    parser.add_argument("--globalNcells", action="store_true")
    parser.add_argument("--globalUcell", action="store_true")
    parser.add_argument("--scaleR1", action="store_true")
    parser.add_argument("--recenter", action="store_true")
    parser.add_argument("--stpmax", default=1e20, type=float)
    parser.add_argument("--usepreoptAmat", action="store_true")
    parser.add_argument("--usepreoptscale", action="store_true")
    parser.add_argument("--usepreoptncells", action="store_true")
    parser.add_argument("--usepreoptbg", action="store_true")
    parser.add_argument("--noprintresbins", action="store_true")

    parser.add_argument("--sad", action="store_true")
    parser.add_argument("--symbol", default="P43212", type=str)
    parser.add_argument("--p9", action="store_true")
    parser.add_argument("--ucellsigma", default=0.005, type=float)
    parser.add_argument("--bgcoefsigma", default=1, type=float)
    parser.add_argument("--ncellssigma", default=0.0005, type=float)
    parser.add_argument("--rotXYZsigma", nargs=3,  default=[0.003, 0.003, 0.001], type=float)
    parser.add_argument("--bgsigma", nargs=3,  default=[0.005, 0.005, 0.01], type=float)
    parser.add_argument("--spotscalesigma",default=0.01, type=float)
    parser.add_argument("--fcellsigma",default=0.005, type=float)
    parser.add_argument("--bs7", action="store_true")
    parser.add_argument("--bs7real", action="store_true")
    parser.add_argument("--loadonly", action="store_true")
    parser.add_argument("--poissononly", action="store_true")
    parser.add_argument("--boopi", type=int, default=0)
    parser.add_argument("--Nmax", type=int, default=-1, help='NOT USING. Max number of images to process per rank')
    parser.add_argument("--nload", type=int, default=None, help='Max number of images to load per rank')
    parser.add_argument("--loadstart", type=int, default=None)
    parser.add_argument("--ngroups", type=int, default=1)
    parser.add_argument("--groupId", type=int, default=0)
    parser.add_argument('--perturblist', default=None, type=int)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--forcemono", action='store_true')
    parser.add_argument("--unknownscale", default=None, type=float, help="Initial scale factor to apply to shots...")
    parser.add_argument("--printallmissets", action='store_true')
    parser.add_argument("--gainrefine", action="store_true")
    parser.add_argument("--fcellbump", default=0.1, type=float)
    parser.add_argument("--initpickle", default=None, type=str, help="path to a pandas pkl file containing optimized parameters")
    parser.add_argument("--oversample", default=0, type=int)
    parser.add_argument("--hack", action="store_true", help="use the local 6 tester files")
    parser.add_argument("--curvatures", action='store_true')
    parser.add_argument("--numposcurvatures", default=7, type=int)
    parser.add_argument("--startwithtruth", action='store_true')
    parser.add_argument("--testmode2", action="store_true", help="debug flag for doing a test run")
    parser.add_argument("--glob", type=str, required=True, help="glob for selecting files (output files of process_mpi")
    parser.add_argument("--partition", action="store_true")
    parser.add_argument("--partitiontime", default=5, type=float, help="seconds allowed for partitioning inputs")
    parser.add_argument("--Fobs", type=str, required=True)
    parser.add_argument("--Fref", type=str, default=None)
    parser.add_argument("--keeperstags", type=str, nargs="+", default=["keepers"], help="names of keepers selection flags")
    parser.add_argument("--plotstats", action="store_true")
    parser.add_argument("--fcellrange", nargs=2,  default=None, type=float, 
        help="2 args specifying lower and upper resolution bounds, then only miller indices within the bound are refined")
    parser.add_argument("--fcell", nargs="+", default=None, type=int)
    parser.add_argument("--ncells", nargs="+", default=None, type=int)
    parser.add_argument("--scale", nargs="+", default=None, type=int)
    parser.add_argument("--umatrix", nargs="+", default=None, type=int)
    parser.add_argument("--bmatrix", nargs="+", default=None, type=int)
    parser.add_argument("--bg", nargs="+", default=None, type=int)
    parser.add_argument("--maxcalls", nargs="+", required=True, type=int)
    parser.add_argument("--plotfcell", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--savepickleonly", action="store_true")
    parser.add_argument("--perturbfcell", default=None, type=float)
    parser.add_argument("--bgextracted", action="store_true")

    args = parser.parse_args()
    print("ARGS:")
    print(args)
    import sys
    print("COMMAND LINE LOOKED LIKE:\n %s" % " ".join(sys.argv))
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
    from numpy import exp as EXP
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
    from cxid9114.geom.multi_panel import CSPAD
    from cctbx.array_family import flex
    flex_double = flex.double
    from cctbx import sgtbx, miller
    from cctbx.crystal import symmetry

    # let the root load the structure factors and energies to later broadcast
    from cxid9114.sf import struct_fact_special
    from cxid9114.parameters import ENERGY_CONV, ENERGY_LOW
    import numpy as np
    LOADTXT = np.loadtxt
    np_log = np.log
    ARANGE = np.arange
    LOGICAL_OR = np.logical_or
    BASENAME = os.path.basename
    # grab the structure factors at the edge energy (ENERGY_LOW=8944 eV)
    ALIST = None
    if args.alist is not None:
        ALIST = list(np.loadtxt(args.alist, str))
    

    from dials.algorithms.indexing.compare_orientation_matrices import difference_rotation_matrix_axis_angle as diff_rot

else:
    ALIST = None
    np_indices = None
    EXP = ARANGE = LOGICAL_OR= LOADTXT = None
    np_log = None
    diff_rot = None
    flex_double = None
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
    BASENAME = None


if has_mpi:
    if rank == 0:
        print("Broadcasting imports")
    ALIST = comm.bcast(ALIST)
    EXP = comm.bcast(EXP)
    LOADTXT = comm.bcast(LOADTXT)
    LOGICAL_OR = comm.bcast(LOGICAL_OR)
    ARANGE = comm.bcast(ARANGE)
    RefineAllMultiPanel = comm.bcast(RefineAllMultiPanel)
    np_indices = comm.bcast(np_indices, root=0)
    np_log = comm.bcast(np_log, root=0)
    glob = comm.bcast(glob, root=0)
    BASENAME = comm.bcast(BASENAME)
    flex_double = comm.bcast(flex_double, root=0)
    diff_rot = comm.bcast(diff_rot, root=0)
    compare_with_ground_truth = comm.bcast(compare_with_ground_truth, root=0)
    args = comm.bcast(args, root=0)
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


class GlobalData:

    def __init__(self):
        self.int_radius = 5  #
        self.gain = args.gainval  # gain of panels, can be refined, can be panel dependent
        self.symbol = args.symbol
        self.anomalous_flag = True
        self.flux_min = 1e2  # minimum number of photons to simulate (assume flux is N-photons, e.g. 1 second exposure)
        self.n_ucell_param = 2  # tetragonal cell
        self.Nload = args.nload  #
        self.all_pix = 0
        self.global_ncells_param = args.globalNcells
        self.global_ucell_param = args.globalUcell
        self.time_load_start = 0
        self.fnames = []  # the filenames containing the datas
        self.all_spot_roi = {}  # one list per shot, rois are x1,x2,y1,y2 per reflection
        self.global_image_id = {}  # gives a  unique ID for each image so multiple ranks can process roi from same image
        self.all_abc_inits = {}  # one list per shot, abc_inits are a,b,c per reflection
        self.all_panel_ids = {}  # one list per shot, panel_ids are single number per reflection
        self.all_ucell_mans = {}  # one per shot, UcellManager instance (Tetragonal in this case)
        self.all_spectra = {}  # one list of (wavelength, flux) tuples per shot
        self.all_crystal_models = {}
        self.all_shot_idx = {}
        self.all_crystal_GT = {}
        self.all_xrel = {}
        self.all_yrel = {}
        self.all_Hi_asu = {}
        self.all_crystal_scales = {}
        self.log_of_init_crystal_scales = {}
        self.all_Hi = {}
        self.all_nanoBragg_rois = {}
        self.SIM = None  # simulator; one per rank!
        self.all_roi_imgs = {}
        self.all_fnames = {}
        self.background_estimate = None
        self.all_proc_idx = {}
        self.all_proc_fnames = {}
        self.m_init = {}
        self.spot_scale_init = {}
        self.nbbeam = self.nbcryst = None
        self.miller_data_map = None

        self.reduced_bbox_keeper_flags = {}
        self.all_bg_coef = {}

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
        self.nbbeam.size_mm = 0.000886226925452758  # NOTE its a circular beam whoops
        #self.nbbeam.size_mm = 0.001
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

        if args.unknownscale is not None:
            #self.SIM.D.spot_scale = 1e6
            self.SIM.D.spot_scale = args.unknownscale
            #self.SIM.D.spot_scale = 15555.1313 kaladin_2k after a small batch starting from some high number
            #self.SIM.D.spot_scale = 17884  # determined from refinement using the syl3 starting model
        else:
            self.SIM.D.spot_scale = 1150
            self.SIM.D.polarization = .999

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
                roi_per += [h5s[i_f]["bboxes"]["shot%d" % (i_shot)].shape[0] 
                            for i_shot in range(Nshots_per_file[i_f])]

            from numpy import array_split
            from numpy.random import permutation
            print ("I am root. Number of uniques = %d" % len(set(shot_tuples)))

            # divide the array into chunks of roughly equal sum (total number of ROI)
            if args.partition and args.restartfile is None and args.xinitfile is None:
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

            # optional to divide into a sub group
            shot_tuples = array_split(shot_tuples, args.ngroups)[args.groupId]
            shots_for_rank = array_split(shot_tuples, size)
            import os  # FIXME, I thought I was imported already!
            if args.outdir is not None:  # save for a fast restart (shot order is important!)
                np.save(os.path.join(args.outdir, "shots_for_rank"), shots_for_rank)
            if args.restartfile is not None:
                # the directory containing the restart file should have a shots for rank file
                dirname = os.path.dirname(args.restartfile)
                print ("Loading shot mapping from dir %s" % dirname)
                shots_for_rank = np.load(os.path.join(dirname, "shots_for_rank.npy"))
                # propagate the shots for rank file...
                if args.outdir is not None:
                    np.save(os.path.join(args.outdir, "shots_for_rank"), shots_for_rank)
            if args.xinitfile is not None:
                # the directory containing the restart file should have a shots for rank file
                dirname = os.path.dirname(args.xinitfile)
                print ("Loading shot mapping from dir %s" % dirname)
                shots_for_rank = np.load(os.path.join(dirname, "shots_for_rank.npy"))
                # propagate the shots for rank file...
                if args.outdir is not None:
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
            if rank==0:
                np.save("shots_for_rank", shots_for_rank)
            shots_for_rank = comm.bcast(shots_for_rank, root=0)
        # h5s = comm.bcast( h5s, root=0)  # pull in the open hdf5 files

        my_shots = shots_for_rank[rank]
        if self.Nload is not None:
            start = 0
            if args.loadstart is not None:
                start = args.loadstart
            my_shots = my_shots[start: start+self.Nload]
        print("Rank %d: I will load %d shots, first shot: %s, last shot: %s"
              % (rank, len(my_shots), my_shots[0], my_shots[-1]))

        # open the unique filenames for this rank
        # TODO: check max allowed pointers to open hdf5 file
        import h5py
        my_unique_fids = set([fidx for fidx, _ in my_shots])
        self.my_open_files = {fidx: h5py_File(self.fnames[fidx], "r") for fidx in my_unique_fids}
        #for fidx in my_unique_fids:
        #    fpath = self.fnames[fidx]
        #    if args.imgdirname is not None:
        #        fpath = fpath.split("/kaladin/")[1]
        #        fpath = os.path.join(args.imgdirname, fpath)
        #    self.my_open_files[fidx] = h5py.File(fpath, "r")
        Ntot = 0

        #self.n_shots = len(my_shots)
        self.n_shots = 0
        img_num = 0
        for iii, (fname_idx, shot_idx) in enumerate(my_shots):
            
            h = self.my_open_files[fname_idx]

            # load the dxtbx image data directly:
            npz_path = h["h5_path"][shot_idx]
            try:
                npz_path = npz_path.decode()
            except AttributeError:
                pass

            if ALIST is not None:
                if BASENAME(npz_path) not in ALIST:
                    continue

            if args.imgdirname is not None:
                import os
                npz_path = npz_path.split("/kaladin/")[1]
                npz_path = os.path.join(args.imgdirname, npz_path)

            #if args.noiseless:
            #    noiseless_path = npz_path.replace(".npz", ".noiseless.npz")
            #    img_handle = numpy_load(noiseless_path)

            #elif args.readoutless:
            #    import os
            #    #readoutless_path = npz_path.split("tang/")[1]
            #    #readoutless_path = os.path.join("/global/project/projectdirs/lcls/dermen/d9114_sims/bear",
            #    #                                readoutless_path)
            #    readoutless_path = npz_path.replace("tang", "bear")
            #    img_handle = numpy_load(readoutless_path)
            else:
                img_handle = numpy_load(npz_path)

            img = img_handle["img"]

            if len(img.shape) == 2:  # if single panel
                img = array([img])

            # D = det_from_dict(img_handle["det"][()])
            B = beam_from_dict(img_handle["beam"][()])
            
            m_init = args.Ncells_size 
            if args.usepreoptncells:
                m_init = h["ncells_%s" % args.preopttag][shot_idx]

            spot_scale_init = 1
            if args.usepreoptscale:
                spot_scale_init = h["spot_scale_%s" % args.preopttag][shot_idx]
            
            Amat = h["Amatrices"][shot_idx]
            if args.usepreoptAmat:
                Amat = h["Amatrices_%s" % args.preopttag][shot_idx]
            amat_elems = list(sqr(Amat).inverse().elems)
            # real space basis vectors:
            a_real = amat_elems[:3]
            b_real = amat_elems[3:6]
            c_real = amat_elems[6:]

            # dxtbx indexed crystal model
            C = Crystal(a_real, b_real, c_real, "P43212")

            # change basis here ? Or maybe just average a/b
            a, b, c, _, _, _ = C.get_unit_cell().parameters()
            a_init = .5 * (a + b)
            c_init = c

            # shoe boxes where we expect spots
            bbox_dset = h["bboxes"]["shot%d" % shot_idx]
            n_bboxes_total = bbox_dset.shape[0]
            # is the shoe box within the resolution ring and does it have significant SNR (see filter_bboxes.py)
            # tilt plane to the background pixels in the shoe boxes
            tilt_abc_dset = h["tilt_abc"]["shot%d" % shot_idx]
            if args.usepreoptbg and not args.bgextracted:
                tilt_abc_dset = h["tilt_abc_%s" % args.preopttag]["shot%d" % shot_idx]

            bg_coef = -1
            if args.bgextracted:
                # if its the first image, load the backgorund estimate array
                if img_num==0:
                    # this should be same length as number of panels in detector e.g. len(CSPAD)
                    self.background_estimate = h["background_estimate"][()] / self.gain
                    
                bg_coef = h["background_coefficients"][shot_idx]
                if args.usepreoptbg:
                    bg_coef = h["background_coefficients_%s" % args.preopttag][shot_idx]

            # miller indices (not yet reduced by symm equivs)
            Hi_dset = h["Hi"]["shot%d" % shot_idx]
            try:
                panel_ids_dset = h["panel_ids"]["shot%d" % shot_idx]
                has_panels = True
            except KeyError:
                has_panels = False

            # BEGIN bbox selection flag management
            # only keep a shoebox if its potentially a keeper
            # Here we provide a means for loading shoeboxes if and only if they
            # will ever be simulated as determined by the keeper flags
            # If a bound box is not flagged anywhere, it will be removed from memory
            bbox_id = ARANGE(n_bboxes_total)
            is_a_keeper = np_zeros(n_bboxes_total).astype(bool)
            kept_bbox_ids = {}
            for keeperstag in set(args.keeperstags):
                keeper_flags = h["bboxes"]["%s%d" % (keeperstag, shot_idx)][()]
                is_a_keeper = LOGICAL_OR(is_a_keeper, keeper_flags)
                kept_bbox_ids[keeperstag] = bbox_id[keeper_flags]
            
            # The following provides a means for selecting the different subsets
            # of ROIs based on the original keeper flags
            all_kept_bbox_ids = bbox_id[is_a_keeper]
            self.reduced_bbox_keeper_flags[img_num] = {}
            for keeperstag in set(args.keeperstags):
                flags = array([i in kept_bbox_ids[keeperstag] for i in all_kept_bbox_ids])
                self.reduced_bbox_keeper_flags[img_num][keeperstag] = flags
            # END selection flag management

            # BEGIN apply the keeper filters:
            bboxes = [bbox_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            tilt_abc = [tilt_abc_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            Hi = [tuple(Hi_dset[i_bb]) for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            proc_file_idx = [i_bb for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            if has_panels:
                panel_ids = [panel_ids_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            else:
                panel_ids = [0] * len(tilt_abc)
            # END apply the keeper filters

            # BEGIN counting pixels
            tot_pix = [(j2 - j1)*(i2 - i1) for i1, i2, j1, j2 in bboxes]
            Ntot += sum(tot_pix)
            # END counting pixels

            # load some ground truth data from the simulation dumps (e.g. spectrum)
            #h5_fname = h["h5_path"][shot_idx].replace(".npz", "")
            h5_fname = npz_path.replace(".npz", "")
            if args.noiseless:
                h5_fname = npz_path.replace(".noiseless.npz", "")
            elif args.readoutless:
                h5_fname = npz_path.replace(".readoutless.npz", "")

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

            fluxes *= es  # multiply by the exposure time
            # TODO: wavelens should come from the imageset file itself
            if "wavelengths" in data.keys():
                wavelens = data["wavelengths"][()]
            else:# elif args.bs7 or args.bs7real:
                from cxid9114.parameters import WAVELEN_HIGH
                wavelens = [WAVELEN_HIGH]

            spectrum = zip(wavelens, fluxes)
            # dont simulate when there are no photons!
            spectrum = [(wave, flux) for wave, flux in spectrum if flux > self.flux_min]
                
            if args.checkbackground:
                true_background = data["background"][()]
                if img_num==0:
                    self.true_residual = np_zeros(true_background.shape)
                    self.true_residual_Nsamples = np_zeros(true_background.shape)

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
                    if args.Fobslabel is not None:
                        self.Fhkl_obs = GlobalData.open_mtz(args.Fobs, args.Fobslabel)
                    else:
                        self.Fhkl_obs = open_flex(args.Fobs).as_amplitude_array()
                    self.Fhkl_ref = args.Fref
                    if args.Fref is not None:
                        if args.Freflabel is not None:
                            self.Fhkl_ref = GlobalData.open_mtz(args.Fref, args.Freflabel)
                        else:
                            self.Fhkl_ref = open_flex(args.Fref).as_amplitude_array()  # this reference miller array is used to track CC and R-factor

                    if args.p9:
                        wavelen = 0.9793
                        #from cxid9114.sf.struct_fact_special import load_p9
                        #Fhkl_guess = load_p9()
                        raise NotImplementedError()
                    elif args.bs7 or args.bs7real:
                        from cxid9114.parameters import WAVELEN_HIGH
                        #from cxid9114.sf import struct_fact_special
                        #import os
                        wavelen = WAVELEN_HIGH
                    else:
                        from cxid9114.parameters import WAVELEN_LOW
                        wavelen = WAVELEN_LOW
                        #from cxid9114.sf.struct_fact_special import load_4bs7_sf
                        #Fhkl_guess = load_4bs7_sf()
                        raise NotImplementedError()

                    if not args.bs7real:
                        spectrum = [(wavelen, fluxes[0])]
                    # end if sad
                self.initialize_simulator(C, B, spectrum, self.Fhkl_obs)

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
            # OLD WAY: 
            #for i_bbox, (_, x2, _, y2) in enumerate(bboxes):
            #    if x2 == nfast:
            #        bboxes[i_bbox][1] = x2 - 1  # update roi_xmax
            #    if y2 == nslow:
            #        bboxes[i_bbox][3] = y2 - 1  # update roi_ymax
            
            # now cache the roi in nanoBragg format ((x1,x2), (y1,y1))
            # and also cache the pixels and the coordinates

            nanoBragg_rois = []  # special nanoBragg format
            xrel, yrel, roi_img = [], [], []
            for i_roi, (x1, x2, y1, y2) in enumerate(bboxes):
                nanoBragg_rois.append(((x1, x2-1), (y1, y2-1)))
                yr, xr = np_indices((y2 - y1, x2 - x1))
                xrel.append(xr)
                yrel.append(yr)
                pid = panel_ids[i_roi]
                sY = slice(y1,y2,1)
                sX = slice(x1,x2,1)
                roi_img.append(img_in_photons[pid, sY, sX])
                if args.checkbackground:
                    pid = panel_ids[i_roi]
                    tx, ty, tz = tilt_abc[i_roi]
                    tilt_plane = tx*xr + ty*yr + tz
                    self.true_residual[pid, sY,sX] = true_background[pid,sY, sX] - tilt_plane
                    self.true_residual_Nsamples[pid, sY, sX] += 1

            # make sure to clear that damn memory
            img = None
            img_in_photons = None
            del img  # not sure if needed here..
            del img_in_photons

            # peak at the memory usage of this rank
            #mem = getrusage(RUSAGE_SELF).ru_maxrss  # peak mem usage in KB
            #mem = mem / 1e6  # convert to GB
            mem = self._usage()

            #print "RANK %d: %.2g total pixels in %d/%d bboxes (file %d / %d); MemUsg=%2.2g GB" \
            #      % (rank, Ntot, len(bboxes), n_bboxes_total,  img_num +1, len(my_shots), mem)
            self.all_pix += Ntot

            # accumulate per-shot information
            self.global_image_id[img_num] = None  # TODO
            self.all_spot_roi[img_num] = bboxes
            self.all_abc_inits[img_num] = tilt_abc
            self.all_panel_ids[img_num] = panel_ids
            self.all_ucell_mans[img_num] = ucell_man
            self.all_spectra[img_num] = spectrum
            self.all_crystal_models[img_num] = C
            self.spot_scale_init[img_num] = spot_scale_init
            self.m_init[img_num] = m_init

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
            self.all_shot_idx[img_num] = shot_idx  # this is the index of the shot in the process*h5 file
            self.all_bg_coef[img_num] = bg_coef

            img_num += 1
            self.n_shots += 1

        for h in self.my_open_files.values():
            h.close()

        #print ("Rank %d; all subimages loaded!" % rank)

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

    def init_global_ucell(self):
        if self.global_ucell_param:
            n_images = len(self.all_spot_roi)
            if args.cella is None and args.cellc is None:
                if rank == 0:
                    print ("Init global ucell without cella and cellc")

                # TODO: implement for non tetragonal
                a_vals, _, c_vals, _, _, _ = zip(*[self.all_crystal_models[i].get_unit_cell().parameters()
                                                   for i in range(n_images)])
                a_vals = list(a_vals)
                c_vals = list(c_vals)
                if has_mpi:
                    a_vals = comm.reduce(a_vals, MPI.SUM, root=0)
                    c_vals = comm.reduce(c_vals, MPI.SUM, root=0)

                a_mean = c_mean = None
                if rank == 0:
                    a_mean = np.median(a_vals)
                    c_mean = np.median(c_vals)
                if has_mpi:
                    a_mean = comm.bcast(a_mean, root=0)
                    c_mean = comm.bcast(c_mean, root=0)
            else:
                a_mean = args.cella
                c_mean = args.cellc

            print ("Rank %d:  Updating ucell mean for each ucell manager to %.4f %.4f" % (rank, a_mean, c_mean))
            for i_shot in range(n_images):
                self.all_ucell_mans[i_shot].variables = a_mean, c_mean

    def tally_statistics(self):

        # tally up all miller indices in this refinement
        self._gather_Hi_information()
        self.num_hkl_global = len(self.idx_from_asu)

        n_images = len(self.all_spot_roi)
        self.n_images = n_images
        n_spot_per_image = [len(self.all_spot_roi[i_image]) for i_image in range(n_images)]
        n_spot_tot = sum(n_spot_per_image)
        total_pix = 0
        for i_image in range(n_images):
            nspot = n_spot_per_image[i_image]
            for x1,x2,y1,y2 in self.all_spot_roi[i_image]:
                total_pix += (x2-x1)*(y2-y1)

        #total_pix = self.all_pix
        # Per image we have 3 rotation angles to refine
        n_rot_param = 3

        # by default we assume each shot refines its own ncells param (mosaic domain size Ncells_abc in nanoBragg)
        n_global_ncells_param = 0
        n_per_image_ncells_param = 1
        if self.global_ncells_param:
            n_global_ncells_param = 1
            n_per_image_ncells_param = 0

        # by default each shot refines its own unit cell parameters (e.g. a,b,c,alpha, beta, gamma)
        n_global_ucell_param = 0
        n_per_image_ucell_param = self.n_ucell_param
        if self.global_ucell_param:
            n_global_ucell_param = self.n_ucell_param
            n_per_image_ucell_param = 0

        # 1 crystal scale factor refined per shot (overall scale)
        n_per_image_scale_param = 1

        # NOTE: n_param_per_image is no longer a constant when we refine background planes
        # NOTE: (unless we do a per-image polynomial fit background plane model)
        
        if not args.bgextracted:
            self.n_param_per_image = [n_rot_param + n_per_image_ncells_param + n_per_image_ucell_param +
                                 n_per_image_scale_param + 3*n_spot_per_image[i]
                                 for i in range(n_images)]
        else:
            self.n_param_per_image = [n_rot_param + n_per_image_ncells_param + n_per_image_ucell_param +
                             n_per_image_scale_param + 1
                             for _ in range(n_images)]

        total_per_image_unknowns = sum(self.n_param_per_image)

        # NOTE: local refers to per-image
        self.n_local_unknowns = total_per_image_unknowns

        mem = self._usage()  # get memory usage
        # note: roi para

        # totals across ranks
        if has_mpi:
            n_images = comm.reduce(n_images, MPI.SUM, root=0)
            n_spot_tot = comm.reduce(n_spot_tot, MPI.SUM, root=0)
            total_pix = comm.reduce(total_pix, MPI.SUM, root=0)
            mem = comm.reduce(mem,MPI.SUM, root=0)

        # Gather so that each rank knows exactly how many local unknowns are on the other ranks
        if has_mpi:
            local_unknowns_per_rank = comm.gather(self.n_local_unknowns)
        else:
            local_unknowns_per_rank = [self.n_local_unknowns]

        if rank == 0:
            total_local_unknowns = sum(local_unknowns_per_rank)  # across all ranks
        else:
            total_local_unknowns = None
        
        self.local_unknowns_across_all_ranks = total_local_unknowns
        if has_mpi:
            self.local_unknowns_across_all_ranks = comm.bcast(self.local_unknowns_across_all_ranks, root=0)

        # TODO: what is the 2 for (its gain and detector distance which are not currently refined...
        self.n_global_params = 2 + n_global_ucell_param + n_global_ncells_param + self.num_hkl_global  # detdist and gain + ucell params

        self.n_total_unknowns = self.local_unknowns_across_all_ranks + self.n_global_params  # gain and detdist (originZ)

        # also report total memory usage
        #mem_tot = mem
        #if has_mpi:
        #    mem_tot = comm.reduce(mem_tot, MPI.SUM, root=0)

        if has_mpi:
            comm.Barrier()
        if rank == 0:
            print("\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
            print("MPIWORLD TOTALZ: images=%d, spots=%d, pixels=%2.2g, Nlocal/Nglboal=%d/%d, usage=%2.2g GigaBytes"
                  % (n_images, n_spot_tot, total_pix, total_local_unknowns,self.n_global_params, mem))
            print("Total time elapsed= %.4f seconds" % (time.time()-self.time_load_start))
            print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")

            # determine where in the global parameter array does this rank
            # parameters begin
            self.starts_per_rank = {}
            xpos = 0
            for _rank, n_unknown in enumerate(local_unknowns_per_rank):
                self.starts_per_rank[_rank] = xpos
                xpos += n_unknown
        else:
            self.starts_per_rank = None

        if has_mpi:
            self.starts_per_rank = comm.bcast(self.starts_per_rank, root=0)

    def _gather_Hi_information(self):
        nshots_on_this_rank = len(self.all_Hi)
        self.Hi_all_ranks, self.Hi_asu_all_ranks = [], []
        for i in range(nshots_on_this_rank):
            self.Hi_all_ranks += self.all_Hi[i]
            self.Hi_asu_all_ranks += self.all_Hi_asu[i]

        #print("Rank %d: Num miller vars on rank=%d" % (comm.rank, len(set(Hasu))))
        if has_mpi:
            self.Hi_all_ranks = comm.reduce(self.Hi_all_ranks, root=0)  # adding python lists concatenates them
            self.Hi_all_ranks = comm.bcast(self.Hi_all_ranks, root=0)
            self.Hi_asu_all_ranks = comm.reduce(self.Hi_asu_all_ranks, root=0)
            self.Hi_asu_all_ranks = comm.bcast(self.Hi_asu_all_ranks, root=0)

        if rank==0:
            from cctbx.array_family import flex as cctbx_flex
            uc = self.all_ucell_mans[0]
            params = uc.a, uc.b, uc.c, uc.al*180/np.pi, uc.be*180/np.pi, uc.ga*180/np.pi
            params = 79.1, 79.1, 38.4, 90,90,90
            symm = symmetry(unit_cell=params, space_group_symbol=self.symbol)
            hi_asu_flex = cctbx_flex.miller_index(self.Hi_asu_all_ranks)
            mset = miller.set(symm, hi_asu_flex, anomalous_flag=True)
            marr = miller.array(mset)# ,data=flex.double(len(hi_asu_felx),0))
            n_bin=10
            binner = marr.setup_binner(d_max=999, d_min=2.125, n_bins=n_bin)
            from collections import Counter
            print("Average multiplicities:")
            print("<><><><><><><><><><><><>")
	    for i_bin in range(n_bin-1):
		dmax, dmin = binner.bin_d_range(i_bin+1)
		F_in_bin = marr.resolution_filter(d_max=dmax, d_min=dmin)
		#multi_data = in_bin.data().as_numpy_array()
                multi_in_bin = array(list(Counter(F_in_bin.indices()).values()))
		print "%2.5g-%2.5g : Multiplicity=%.4f" % (dmax, dmin,multi_in_bin.mean()  )
                for ii in range(1,100,8):
                    print("\t %d refls with multi %d" % (sum(multi_in_bin==ii), ii))
        
            print("Overall completeness\n<><><><><><><><>")
            symm = symmetry(unit_cell=params, space_group_symbol=self.symbol)
            hi_flex_unique = cctbx_flex.miller_index(list(set(self.Hi_asu_all_ranks)))
            mset = miller.set(symm, hi_flex_unique, anomalous_flag=True)
            binner = mset.setup_binner(d_min=2.125, d_max=999, n_bins=10)
            mset.completeness(use_binning=True).show()
            marr_unique_h = miller.array(mset)# ,data=flex.double(len(hi_asu_felx),0))
            print("Rank %d: total miller vars=%d" % (rank, len(set(self.Hi_asu_all_ranks))))
        if rank > 0:
            marr_unique_h = None

        if has_mpi:
            marr_unique_h = comm.bcast(marr_unique_h)

        # this will map the measured miller indices to their index in the LBFGS parameter array self.x
        self.idx_from_asu = {h: i for i, h in enumerate(set(self.Hi_asu_all_ranks))}
        # we will need the inverse map during refinement to update the miller array in diffBragg, so we cache it here
        self.asu_from_idx = {i: h for i, h in enumerate(set(self.Hi_asu_all_ranks))}

        fres = marr_unique_h.d_spacings()
        self.res_from_asu = {h:res for h,res in zip(fres.indices(), fres.data())}
        # will we only refine a range of miller indices ? 
        self.freeze_idx = None
        if args.fcellrange is not None:
            self.freeze_idx = {}
            resmax, resmin = args.fcellrange
            for h in self.idx_from_asu:
                res = self.res_from_asu[h]
                if res >= resmin  and res < resmax:
                    self.freeze_idx[h] = False
                else:
                    self.freeze_idx[h] = True

    def pre_refine_setup(self, i_trial=0, refine_fcell=None, refine_spot_scale=None, refine_Umatrix=None, 
            refine_Bmatrix=None, refine_ncells=None, refine_bg=None, max_calls=None, x_init=None):

        self.RUC = GlobalRefiner(
            n_total_params=self.n_total_unknowns,
            n_local_params=self.n_local_unknowns,
            n_global_params=self.n_global_params,
            local_idx_start=self.starts_per_rank[rank],
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
            perturb_fcell=args.perturbfcell,
            global_ncells=args.globalNcells, global_ucell=args.globalUcell,
            shot_originZ_init= {img_num:CSPAD[0].get_origin()[2] for img_num in range(self.n_shots)},
            shot_bg_coef=self.all_bg_coef, background_estimate=self.background_estimate)
        
        self.i_trial = i_trial

        if refine_Bmatrix is not None:
            self.RUC.refine_Bmatrix = refine_Bmatrix 
        if refine_Umatrix is not None:
            self.RUC.refine_Umatrix = refine_Umatrix 
        if refine_fcell is not None:
            self.RUC.refine_Fcell = refine_fcell 
        if refine_ncells is not None:
            self.RUC.refine_ncells = refine_ncells 
        if refine_bg is not None:
            self.RUC.refine_background_planes = refine_bg
        if refine_spot_scale is not None:
            self.RUC.refine_crystal_scale = refine_spot_scale
        if max_calls is not None:
            self.RUC.max_calls = max_calls 

        self.RUC.x_init = x_init
        self.RUC.only_pass_refined_x_to_lbfgs = args.xrefinedonly
        self.RUC.bg_extracted = args.bgextracted
        
        self.RUC.recenter = args.recenter
        # parameter rescaling...
        self.RUC.rescale_params = True
        self.RUC.rescale_fcell_by_resolution = not args.NoRescaleFcellRes
        
        self.RUC.spot_scale_init = self.spot_scale_init 
        self.RUC.m_init = self.m_init  

        self.RUC.ignore_line_search_failed_step_at_lower_bound = args.ignorelinelow
        #FIXME 
        self.RUC.ucell_inits = [self.all_ucell_mans[i_shot].variables for i_shot in range(self.n_shots)]
        #FIXME
        self.RUC.rotX_sigma = args.rotXYZsigma[0]
        self.RUC.rotY_sigma = args.rotXYZsigma[1]
        self.RUC.rotZ_sigma = args.rotXYZsigma[2]
        self.RUC.ucell_sigmas = [args.ucellsigma, args.ucellsigma]
        self.RUC.bg_coef_sigma = args.bgcoefsigma
        self.RUC.originZ_sigma = 1  # 0.01
        self.RUC.m_sigma = args.ncellssigma
        self.RUC.spot_scale_sigma = args.spotscalesigma  # stage1/2.01
        asig, bsig, csig = args.bgsigma
        self.RUC.a_sigma = asig # 0.005
        self.RUC.b_sigma = bsig #0.005
        self.RUC.c_sigma = csig #0.01
        self.RUC.fcell_sigma_scale = args.fcellsigma #0.005
        self.RUC.fcell_resolution_bin_Id = None
        self.RUC.compute_image_model_correlation = args.imagecorr
        # end of parameter rescaling

        # plot things
        self.RUC.sigma_r = 3./args.gainval
        self.RUC.gradient_only=args.gradientonly
        self.RUC.fix_params_with_negative_curvature = args.forcecurva
        #self.RUC.stpmax = args.stpmax
        self.RUC.debug = args.debug
        self.RUC.binner_dmax = 999
        self.RUC.binner_dmin = 2.1
        self.RUC.binner_nbin = 10
        self.RUC.trial_id = self.i_trial
        self.RUC.print_all_missets = args.printallmissets
        self.RUC.print_all_corr = False
        self.RUC.Fref = self.Fhkl_ref
        self.RUC.merge_stat_frequency=3
        self.RUC.min_multiplicity=args.minmulti
        self.RUC.print_resolution_bins= not args.noprintresbins
        self.RUC.refine_rotZ = not args.fixrotZ
        self.RUC.plot_images = args.plot
        self.RUC.plot_fcell = args.plotfcell
        self.RUC.plot_residuals = args.residual
        self.RUC.plot_statistics = args.plotstats
        self.RUC.setup_plots()

        self.RUC.log_fcells = True
        self.RUC.big_dump = args.bigdump

        self.RUC.idx_from_asu = self.idx_from_asu
        self.RUC.asu_from_idx = self.asu_from_idx
        self.RUC.freeze_idx = self.freeze_idx
        self.RUC.scale_r1 = True
        self.RUC.request_diag_once = False
        self.RUC.S = self.SIM
        self.RUC.restart_file = args.restartfile
        self.RUC.has_pre_cached_roi_data = True
        self.RUC.trad_conv = True
        self.RUC.fcell_bump = args.fcellbump
        self.RUC.refine_detdist = False
        self.RUC.S.D.update_oversample_during_refinement = False
        self.RUC.refine_gain_fac = False
        self.RUC.use_curvatures = args.forcecurva
        self.RUC.use_curvatures_threshold = args.numposcurvatures
        if not args.curvatures:
            self.RUC.S.D.compute_curvatures=False
        self.RUC.calc_curvatures = args.curvatures
        self.RUC.poisson_only = args.poissononly
        self.RUC.plot_stride = args.stride
        self.RUC.trad_conv_eps = args.tradeps #5e-10  # NOTE this is for single panel model
        self.RUC.verbose = False
        self.RUC.use_rot_priors = False
        self.RUC.use_ucell_priors = False
        self.RUC.filter_bad_shots = args.filterbad
        #TODO optional properties.. make this obvious
        self.RUC.FNAMES = self.all_fnames
        self.RUC.PROC_FNAMES = self.all_proc_fnames
        self.RUC.PROC_IDX = self.all_shot_idx
        self.RUC.BBOX_IDX = self.all_proc_idx

        self.RUC.Hi = self.all_Hi
        self.RUC.output_dir = args.outdir
        if args.verbose:
            if rank == 0:  # only show refinement stats for rank 0
                self.RUC.verbose = True
        self.RUC.run(setup_only=True)

    def refine(self, selection_flags=None):
        self.RUC.num_positive_curvatures = 0
        self.RUC.use_curvatures = args.forcecurva
        self.RUC.hit_break_to_use_curvatures = False
        self.RUC.selection_flags = selection_flags
        
        if args.tryscipy:
            self.RUC.calc_curvatures = False
            #self.RUC._setup()
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
                                fprime=fprime, args=[self.RUC], factr=args.scipyfactr)

        else:
            self.RUC.run(setup=False)
            if self.RUC.hit_break_to_use_curvatures:
                self.RUC.fix_params_with_negative_curvature = False
                self.RUC.num_positive_curvatures = 0
                self.RUC.use_curvatures = True
                self.RUC.run(setup=False)


    def save_lbfgs_x_array_as_dataframe(self, outname):
        # Here we can save the refined parameters
        my_shots = self.all_shot_idx.keys()
        x = self.RUC.Xall
        data_to_send = []
        image_corr = self.RUC.image_corr
        if image_corr is None:
            image_corr = [-1]*len(my_shots)
        for i_shot in my_shots:
            rotX = self.RUC._get_rotX(i_shot)
            rotY = self.RUC._get_rotY(i_shot)
            rotZ = self.RUC._get_rotZ(i_shot)
            if not args.savepickleonly:
                ang, ax = self.RUC.get_correction_misset(as_axis_angle_deg=True, i_shot=i_shot)
                Bmat = self.RUC.get_refined_Bmatrix(i_shot)
            else:
                ang,ax = self.RUC.get_correction_misset(as_axis_angle_deg=True, anglesXYZ = (rotX, rotY, rotZ))
                pars = self.RUC._get_ucell_vars(i_shot)
                self.all_ucell_mans[i_shot].variables = pars
                Bmat = self.all_ucell_mans[i_shot].B_recipspace 

            bg_coef = -1
            if args.bgextracted:
                bg_coef = self.RUC._get_bg_coef(i_shot)
            
            C = self.RUC.CRYSTAL_MODELS[i_shot]
            C.set_B(Bmat)
            try:
                C.rotate_around_origin(ax, ang)
            except RuntimeError:
                pass
#############################################
            if args.savepickleonly:
                a_init, _, c_init, _, _, _ = self.all_crystal_models[i_shot].get_unit_cell().parameters()
                a_tru, b_tru, c_tru = self.all_crystal_GT[i_shot].get_real_space_vectors()
                try:
                    final_misori = compare_with_ground_truth(a_tru, b_tru, c_tru,[C],symbol="P43212")[0]
                except Exception as err:
                    final_misori = -1
###############################

            Amat_refined = C.get_A()
            ucell_a,_,ucell_c,_,_,_ = C.get_unit_cell().parameters()

            fcell_xstart = self.RUC.fcell_xstart
            ucell_xstart = self.RUC.ucell_xstart[i_shot]
            scale_xpos = self.RUC.spot_scale_xpos[i_shot]
            ncells_xpos = self.RUC.ncells_xpos[i_shot]
            nspots = len(self.RUC.NANOBRAGG_ROIS[i_shot])
           
            bgplane_xpos = -1 
            bgplane = 0
            if not args.bgextracted:
                bgplane_a_xpos = [self.RUC.bg_a_xstart[i_shot][i_spot] for i_spot in range(nspots)]
                bgplane_b_xpos = [self.RUC.bg_b_xstart[i_shot][i_spot] for i_spot in range(nspots)]
                bgplane_c_xpos = [self.RUC.bg_c_xstart[i_shot][i_spot] for i_spot in range(nspots)]
                bgplane_xpos = list(zip(bgplane_a_xpos, bgplane_b_xpos, bgplane_c_xpos))
                bgplane = [self.RUC._get_bg_vals(i_shot, i_spot) for i_spot in range(nspots)]

            crystal_scale = self.RUC._get_spot_scale(i_shot)
            proc_h5_fname = self.all_proc_fnames[i_shot]
            proc_h5_idx = self.all_shot_idx[i_shot]
            proc_bbox_idx = self.all_proc_idx[i_shot]

            ncells_val = self.RUC._get_m_val(i_shot)
            if not args.savepickleonly: 
                init_misori = self.RUC.get_init_misorientation(i_shot)
                final_misori = self.RUC.get_current_misorientation(i_shot)
                img_corr= self.RUC._get_image_correlation(i_shot)
                init_img_corr = self.RUC._get_init_image_correlation(i_shot)
            else:
                init_misori = self.init_misori[i_shot]
                #final_misori = self.init_misori[i_shot]  # computed above
                img_corr = -1
                init_img_corr = -1
            
            data_to_send.append((proc_h5_fname, proc_h5_idx, proc_bbox_idx,crystal_scale, Amat_refined, ncells_val, bgplane, \
                img_corr, init_img_corr, fcell_xstart, ucell_xstart, rotX, rotY, rotZ, scale_xpos, \
                ncells_xpos, bgplane_xpos, init_misori, final_misori, ucell_a, ucell_c, bg_coef))
        
        if has_mpi:
            data_to_send = comm.reduce(data_to_send, MPI.SUM, root=0)
        if rank == 0:
            import pandas
            import h5py
            fnames, shot_idx, bbox_idx, xtal_scales, Amats, ncells_vals, bgplanes, image_corr, init_img_corr, \
                fcell_xstart, ucell_xstart, rotX, rotY, rotZ, scale_xpos, ncells_xpos, bgplane_xpos, \
                init_misori, final_misori, ucell_a, ucell_c, bg_coef = zip(*data_to_send)

            df = pandas.DataFrame({"proc_fnames": fnames, "proc_shot_idx": shot_idx, "bbox_idx": bbox_idx,
                                   "spot_scales": xtal_scales, "Amats": Amats, "ncells": ncells_vals,
                                   "bgplanes": bgplanes, "image_corr": image_corr,
                                   "init_image_corr": init_img_corr,
                                   "fcell_xstart": fcell_xstart,
                                   "ucell_xstart": ucell_xstart,
                                   "init_misorient": init_misori, "final_misorient": final_misori,
                                   "bg_coef": bg_coef,
                                   "rotX": rotX,
                                   "rotY": rotY,
                                   "rotZ": rotZ,
                                   "a": ucell_a, "c": ucell_c, 
                                   "scale_xpos": scale_xpos,
                                   "ncells_xpos": ncells_xpos,
                                   "bgplanes_xpos": bgplane_xpos})
            u_fnames = df.proc_fnames.unique()

            u_h5s = {f:h5py.File(f,'r')["h5_path"][()] for f in u_fnames}
            img_fnames = []
            for f,idx in df[['proc_fnames','proc_shot_idx']].values:
                img_fnames.append( u_h5s[f][idx] )
            df["imgpaths"] = img_fnames

            df.to_pickle(outname)

    def init_misset_results(self):
        results = []
        self.init_misori = {}
        nshots = len(self.all_crystal_GT)
        for i_shot in range(nshots): # (angx, angy, angz, a,c) in enumerate(zip(rotx, roty, rotz, avals, cvals)):

            a_init, _, c_init, _, _, _ = self.all_crystal_models[i_shot].get_unit_cell().parameters()
            a_tru, b_tru, c_tru = self.all_crystal_GT[i_shot].get_real_space_vectors()
            try:
                angular_offset_init = compare_with_ground_truth(a_tru, b_tru, c_tru,
                                                [self.all_crystal_models[i_shot]],
                                                symbol="P43212")[0]
            except Exception as err:
                print("Rank %d img %d err %s" % (rank, i_shot, err))
                angular_offset_init = -1
            results.append(angular_offset_init)
            self.init_misori[i_shot] = angular_offset_init
        return results

    #TODO: test this method ;)
    @staticmethod
    def open_mtz(mtzfname, mtzlabel=None):
        if mtzlabel is None:
            mtzlabel = "fobs(+)fobs(-)"
        print ("Opening mtz file %s" % mtzfname)
        from iotbx.reflection_file_reader import any_reflection_file
        miller_arrays = any_reflection_file(mtzfname).as_miller_arrays()

        possible_labels = []
        foundlabel = False
        for ma in miller_arrays:
            label = ma.info().label_string()
            possible_labels.append(label)
            if label == mtzlabel:
                foundlabel = True
                break

        assert foundlabel, "MTZ Label not found... \npossible choices: %s" % (" ".join(possible_labels))
        ma = ma.as_amplitude_array()
        return ma

    def write_residual_background_image(self):
        assert args.checkbackground
        if has_mpi:
	    self.true_residual = comm.reduce(self.true_residual) 
	    self.true_residual_Nsamples = comm.reduce(self.true_residual_Nsamples)
        if rank==0:
            RES = self.true_residual / self.true_residual_Nsamples
            from numpy import nan_to_num
            RES = nan_to_num(RES)
            from numpy import abs as numpy_abs
            ABS_RES = numpy_abs(RES)
            from dxtbx.model.beam import BeamFactory
            from cxid9114.parameters import WAVELEN_HIGH
            beam = BeamFactory.simple(WAVELEN_HIGH)
            np.savez(args.checkbackgroundsavename, img=RES, det=CSPAD.to_dict(), beam=beam.to_dict())
            np.savez(args.checkbackgroundsavename+ "_abs", img=ABS_RES, det=CSPAD.to_dict(), beam=beam.to_dict())
            print("Saved background residual image to file %s" % args.checkbackgroundsavename)
        if has_mpi:
            comm.Barrier()
        exit()
####pr = cProfile.Profile()
####pr.enable()

##############
# LOAD STAGE #
##############

B = GlobalData()
fnames = glob(args.glob)
B.fnames = fnames
B.load()
print("Finished with the load!")

ang_res = B.init_misset_results()
if has_mpi:
    ang_res = comm.reduce(ang_res, MPI.SUM, root=0)
if rank == 0:
    miss = [a for a in ang_res if a > 0]

    print("INIT MISSETS\n%s"%", ".join(map(str,ang_res)))
    print("INITIAL MEDIAN misset = %f" % np.median(miss))
    print("INITIAL MAX misset = %f" % np.max(miss))
    print("INITIAL MIN misset = %f" % np.min(miss))
if has_mpi:
    comm.Barrier()
if args.checkbackground:
    B.write_residual_background_image()
B.tally_statistics()
B.init_global_ucell()
if has_mpi:
    comm.Barrier()

trials = {"fcell": args.fcell,
          "scale": args.scale,
          "umatrix": args.umatrix,
          "bmatrix": args.bmatrix,
          "ncells": args.ncells,
          "bg": args.bg,
          "max_calls": args.maxcalls}

Ntrials = len(trials["max_calls"])

################
# SETUP/REFINE #
################
x_init = None
if args.xinitfile is not None:
    x_init = flex_double(np_load(args.xinitfile)["x"])
if rank==0:
    import time

if args.protocol=="per_shot":

    for i_trial in range(Ntrials): 
        if rank==0:
            tstart = time.time()
        setup_args = {"max_calls": args.maxcalls[i_trial],
                "refine_fcell": bool(args.fcell[i_trial]),
                "refine_Umatrix": bool(args.umatrix[i_trial]),
                "refine_Bmatrix": bool(args.bmatrix[i_trial]),
                "refine_ncells": bool(args.ncells[i_trial]),
                "refine_bg": bool(args.bg[i_trial]),
                "refine_spot_scale": bool(args.scale[i_trial]),
                "i_trial": i_trial, 
                "x_init": x_init}
        B.pre_refine_setup(**setup_args) 
        #TODO MPI select for global_refiner.py 
        keeperstag = args.keeperstags[i_trial] 
        for i_shot in range(B.n_shots):
            flags = {i_shot: B.reduced_bbox_keeper_flags[i_shot][keeperstag]}
            B.refine(selection_flags = flags)
            if rank == 0:
                print ("<><><><><><><><><><><><><><><><><><><><><><><>")
                print ("<><><> END OF TRIAL %02d ; shot %d/%d <><><><>" % (i_trial+1, i_shot+1, B.n_shots))
                print ("<><><><><><><><><><><><><><><><><><><><><><><>")
        if has_mpi:
            comm.Barrier()
        x_init = B.RUC.Xall

        if rank==0:
            tdone = time.time()-tstart
            print("TRIAL %d TIMEINGZ = %f secz" % (i_trial+1, tdone))
        outname = "%s_trial%d.pkl" % (args.optoutname, i_trial+1)
        B.save_lbfgs_x_array_as_dataframe(outname)

elif args.protocol == "global":
    for i_trial in range(Ntrials): 
        if rank==0:
            tstart = time.time()
        setup_args = {"max_calls": args.maxcalls[i_trial],
                "refine_fcell": bool(args.fcell[i_trial]),
                "refine_Umatrix": bool(args.umatrix[i_trial]),
                "refine_Bmatrix": bool(args.bmatrix[i_trial]),
                "refine_ncells": bool(args.ncells[i_trial]),
                "refine_bg": bool(args.bg[i_trial]),
                "refine_spot_scale": bool(args.scale[i_trial]),
                "i_trial": i_trial, 
                "x_init": x_init}
        B.pre_refine_setup(**setup_args) 
        if not args.savepickleonly:
            B.refine() 
        if rank == 0:
            print ("<><><><><><><><><><><><><><><><><><><><><><><>")
            print ("<><><> END OF TRIAL %02d ;  <><><><>" % (i_trial+1))
            print ("<><><><><><><><><><><><><><><><><><><><><><><>")
        if has_mpi:
            comm.Barrier()
        x_init = B.RUC.Xall

        if rank==0:
            tdone = time.time()-tstart
            print("TRIAL %d TIMEINGZ = %f secz" % (i_trial+1, tdone))
        outname = "%s_trial%d.pkl" % (args.optoutname, i_trial+1)
        B.save_lbfgs_x_array_as_dataframe(outname)


#proc_fnames_shots = [(B.all_proc_fnames[i], B.all_shot_idx[i]) for i in my_shots]

#parameters =[
#    (f, i, np.exp(x[B.RUC.spot_scale_xpos[i]]), x[B.RUC.rotX_xpos[i]], x[B.RUC.rotY_xpos[i]], x[B.RUC.rotZ_xpos[i]])
#    for f, i in proc_fnames_shots]

#pr.disable()
#
#pr.dump_stats('cpu_%d.prof' %comm.rank)
## - for text dump
#with open( 'cpu_%d.txt' %comm.rank, 'w') as output_file:
#    sys.stdout = output_file
#    pr.print_stats(sort='time')
#    sys.stdout = sys.__stdout__

#comm.Barrier()
#B.print_results()

