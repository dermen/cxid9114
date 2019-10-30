

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
    
from dxtbx.model.detector import DetectorFactory
det_from_dict = DetectorFactory.from_dict
from dxtbx.model.beam import BeamFactory
beam_from_dict = BeamFactory.from_dict

# import functions on rank 0 only
if rank == 0:
    from argparse import ArgumentParser
    parser = ArgumentParser("Load and refine bigz")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--logdir", type=str, default='.', help="where to write log files (one per rank)")
    parser.add_argument("--stride", type=int, default=10, help='plot stride')
    parser.add_argument("--residual", action='store_true')
    parser.add_argument("--Nmax", type=int, default=-1, help='Max number of images to process per rank')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--curvatures", action='store_true')
    parser.add_argument("--sad", action='store_true')
    #parser.add_argument("--truthdet", action="store_true")
    parser.add_argument("--startwithtruth", action='store_true')
    parser.add_argument("--glob", type=str, required=True, help="glob for selecting files (output files of process_mpi")
    parser.add_argument("--testmode", action="store_true", help="debug flag for doing a test run")
    parser.add_argument("--testmode2", action="store_true", help="debug flag for doing a test run")
    parser.add_argument("--keeperstag", type=str, default="keepers", help="name of keepers boolean array")
    parser.add_argument("--split", action="store_true", help="split evaluation on a single rank")
    args = parser.parse_args()
    from h5py import File as h5py_File
    from cxid9114.integrate.integrate_utils import Integrator
    from numpy import array, sqrt, percentile
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
    #from simtbx.diffBragg.refiners import RefineAll as RefineAllMultiPanel
    if args.startwithtruth:
        from cxid9114.geom.multi_panel import CSPAD
    else:
        from cxid9114.geom.multi_panel import CSPAD_refined as CSPAD

    # let the root load the structure factors and energies to later broadcast
    from cxid9114.sf import struct_fact_special
    import os
    sf_path = os.path.dirname(struct_fact_special.__file__)
    sfall_file = os.path.join(sf_path, "realspec_sfall.h5")
    data_sf, data_energies = struct_fact_special.load_sfall(sfall_file)
    from cxid9114.parameters import ENERGY_CONV, ENERGY_LOW
    import numpy as np
    # grab the structure factors at the edge energy (ENERGY_LOW=8944 eV)
    edge_idx = np.abs(data_energies-ENERGY_LOW).argmin()
    Fhkl_guess = data_sf[edge_idx]

    if args.sad:
        sad_range = range(edge_idx-10, edge_idx+10)
        data_energies = data_energies[sad_range]

    wavelens = ENERGY_CONV / data_energies

    from dials.algorithms.indexing.compare_orientation_matrices import difference_rotation_matrix_axis_angle as diff_rot

else:
    diff_rot = None
    compare_with_ground_truth = None
    args = None
    RefineAllMultiPanel = None
    nanoBragg_beam = nanoBragg_crystal = None
    SimData = None
    #beam_from_dict = det_from_dict = None
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
    glob = comm.bcast(glob)
    diff_rot = comm.bcast(diff_rot)
    compare_with_ground_truth = comm.bcast(compare_with_ground_truth, root=0)
    args = comm.bcast(args, root=0)
    RefineAllMultiPanel = comm.bcast(RefineAllMultiPanel, root=0)
    Fhkl_guess = comm.bcast(Fhkl_guess, root=0)
    #if args.startwithtruth and args.sad:
    if args.sad:
        from cxid9114.sf.struct_fact_special import load_4bs7_sf
        Fhkl_guess = load_4bs7_sf()
    wavelens = comm.bcast(wavelens, root=0)
    Crystal = comm.bcast(Crystal, root=0)
    sqr = comm.bcast(sqr, root=0)
    CSPAD = comm.bcast(CSPAD, root=0)
    nanoBragg_beam = comm.bcast(nanoBragg_beam, root=0)
    nanoBragg_crystal = comm.bcast(nanoBragg_crystal, root=0)
    SimData = comm.bcast(SimData, root=0)
    #beam_from_dict = comm.bcast(beam_from_dict, root=0)
    #det_from_dict = comm.bcast(det_from_dict, root=0)
    h5py_File = comm.bcast(h5py_File, root=0)
    Integrator = comm.bcast(Integrator, root=0)
    array = comm.bcast(array, root=0)
    sqrt = comm.bcast(sqrt, root=0)
    percentile = comm.bcast(percentile, root=0)
    np_zeros = comm.bcast(np_zeros, root=0)
    np_sum = comm.bcast(np_sum, root=0)
    Process = comm.bcast(Process, root=0)
    getpid = comm.bcast( getpid, root=0)
    numpy_load = comm.bcast(numpy_load, root=0)
    getrusage = comm.bcast(getrusage, root=0)
    RUSAGE_SELF = comm.bcast(RUSAGE_SELF, root=0)
    TetragonalManager = comm.bcast(TetragonalManager, root=0)


class BigData:

    def __init__(self):
        self.int_radius = 5
        self.gain = 28
        self.flux_min = 1e2
        self.Nload = None
        self.fnames = []

    #@profile
    def load(self):

        # some parameters

        # NOTE: for reference, inside each h5 file there is 
        #   [u'Amatrices', u'Hi', u'bboxes', u'h5_path']
        
        # get the total number of shots using worker 0
        if rank == 0:
            print("I am root. I am calculating total number of shots")
            h5s = [h5py_File(f, "r") for f in self.fnames]
            Nshots_per_file = [ h["h5_path"].shape[0] for h in h5s]
            Nshots_tot = sum(Nshots_per_file)
            print("I am root. Total number of shots is %d" % Nshots_tot)

            print("I am root. I will divide shots amongst workers.")
            shot_tuples = []
            for i_f, fname in enumerate(self.fnames):
                fidx_shotidx = [(i_f, i_shot) for i_shot in range(Nshots_per_file[i_f])] 
                shot_tuples += fidx_shotidx

            from numpy import array_split
            print ("I am root. Number of uniques = %d" % len(set(shot_tuples)))
            shots_for_rank = array_split(shot_tuples, size)

            # close the open h5s.. 
            for h in h5s:
                h.close()

        else:
            Nshots_tot = None
            shots_for_rank = None
            h5s = None
        
        #Nshots_tot = comm.bcast( Nshots_tot, root=0)
        if has_mpi:
            shots_for_rank = comm.bcast(shots_for_rank, root=0)
        #h5s = comm.bcast( h5s, root=0)  # pull in the open hdf5 files
        
        my_shots = shots_for_rank[rank]
        if self.Nload is not None:
            my_shots = my_shots[:self.Nload]

        # open the unique filenames for this rank
        # TODO: check max allowed pointers to open hdf5 file
        my_unique_fids = set([fidx for fidx, _ in my_shots])
        my_open_files = {fidx: h5py_File(self.fnames[fidx], "r") for fidx in my_unique_fids}
        Ntot = 0
        self.all_bbox_pixels = []
        for img_num, (fname_idx, shot_idx) in  enumerate(my_shots):
            if img_num < 3:
                continue
            if img_num == args.Nmax:
                #print("Already processed maximum number images!")
                continue
            h = my_open_files[fname_idx]
           
            # load the dxtbx image data directly:
            npz_path = h["h5_path"][shot_idx]
            # NOTE take me out!
            if args.testmode:
                import os
                npz_path = os.path.basename(npz_path)
                npz_path ="cspad_rank0_data25.h5.npz"
            if args.testmode2:
                import os
                npz_path = npz_path.split("d9114_sims/")[1]
                npz_path = os.path.join("/Users/dermen/", npz_path)
            img_handle = numpy_load(npz_path)
            img = img_handle["img"]

            if len(img.shape) == 2:  # if single panel>>
                img = np.array([img])

            #D = det_from_dict(img_handle["det"][()])
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
            
            # change basis here ? Or maybe just average a/b
            a, b, c, _, _, _ = C.get_unit_cell().parameters()
            a_init = .5*(a+b)
            c_init = c

            # shoe boxes where we expect spots
            bbox_dset = h["bboxes"]["shot%d" % shot_idx]
            n_bboxes_total = bbox_dset.shape[0] 
            # is the shoe box within the resolution ring and does it have significant SNR (see filter_bboxes.py)
            is_a_keeper = h["bboxes"]["keepers%d" % shot_idx][()]

            # tilt plane to the background pixels in the shoe boxes
            tilt_abc_dset = h["tilt_abc"]["shot%d" % shot_idx]
            try:
                panel_ids_dset = h["panel_ids"]["shot%d" % shot_idx]
                has_panels = True
            except KeyError:
                has_panels = False

            # apply the filters:
            bboxes = [bbox_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            tilt_abc = [tilt_abc_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            if has_panels:
                panel_ids = [panel_ids_dset[i_bb] for i_bb in range(n_bboxes_total) if is_a_keeper[i_bb]]
            else:
                panel_ids = [0]*len(tilt_abc)

            # how many pixels do we have
            tot_pix = [(j2-j1)*(i2-i1) for i1, i2, j1, j2 in bboxes]
            Ntot += sum(tot_pix)

            # actually load the pixels...  
            #data_boxes = [ img[j1:j2, i1:i2] for i1,i2,j1,j2 in bboxes]

            # Here we will try a per-shot refinement of the unit cell and Umatrix, as well as ncells abc
            # and spot scale etc..

            # load some ground truth data from the simulation dumps (e.g. spectrum)
            h5_fname = h["h5_path"][shot_idx].replace(".npz", "")
            # NOTE remove me
            if args.testmode:
                h5_fname = os.path.basename(h5_fname)
                h5_fname = "cspad_rank0_data25.h5"
            if args.testmode2:
                h5_fname = npz_path.split(".npz")[0]
            data = h5py_File(h5_fname, "r")

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
            fluxes *= es  # multiply by the exposure time
            spectrum = zip(wavelens, fluxes)
            # dont simulate when there are no photons!
            spectrum = [(wave, flux) for wave, flux in spectrum if flux > self.flux_min]

            # make a unit cell manager that the refiner will use to track the B-matrix
            aa, _, cc, _, _, _ = C_tru.get_unit_cell().parameters()
            ucell_man = TetragonalManager(a=a_init, c=c_init)
            if args.startwithtruth:
                ucell_man = TetragonalManager(a=aa, c=cc)

            # create the sim_data instance that the refiner will use to run diffBragg
            # create a nanoBragg crystal
            nbcryst = nanoBragg_crystal()
            nbcryst.dxtbx_crystal = C
            if args.startwithtruth:
                nbcryst.dxtbx_crystal = C_tru
            nbcryst.thick_mm = 0.1
            nbcryst.Ncells_abc = 30, 30, 30
            nbcryst.miller_array = Fhkl_guess.as_amplitude_array()
            nbcryst.n_mos_domains = 1
            nbcryst.mos_spread_deg = 0.0

            # create a nanoBragg beam
            nbbeam = nanoBragg_beam()
            nbbeam.size_mm = 0.001
            nbbeam.unit_s0 = B.get_unit_s0()
            nbbeam.spectrum = spectrum

            # sim data instance
            SIM = SimData()
            SIM.detector = CSPAD
            #SIM.detector = D
            SIM.crystal = nbcryst
            SIM.beam = nbbeam
            SIM.panel_id = 0  # default

            spot_scale = 12
            if args.sad:
                spot_scale = 1
            SIM.instantiate_diffBragg(default_F=0, oversample=0)
            SIM.D.spot_scale = spot_scale

            img_in_photons = img / self.gain
            print("Rank %d, Starting refinement!" % rank)
            try:
                n_spot = 20
                bboxes = array(bboxes)[:n_spot]
                tilt_abc = array(tilt_abc)[:n_spot]
                RUC = RefineAllMultiPanel(
                    spot_rois=bboxes,
                    abc_init=tilt_abc,
                    img=img_in_photons,  # NOTE this is now a multi panel image
                    SimData_instance=SIM,
                    plot_images=args.plot,
                    plot_residuals=args.residual,
                    ucell_manager=ucell_man)

                RUC.panel_ids = panel_ids
                RUC.multi_panel = True
                RUC.split_evaluation = args.split
                RUC.trad_conv = True
                RUC.refine_detdist = False
                RUC.refine_background_planes = False
                RUC.refine_Umatrix = True
                RUC.refine_Bmatrix = True
                RUC.refine_ncells = True
                RUC.use_curvatures = False  # args.curvatures
                RUC.calc_curvatures = True  #args.curvatures
                RUC.refine_crystal_scale = True
                RUC.use_curvatures_threshold = 3
                RUC.refine_gain_fac = False
                RUC.plot_stride = args.stride
                RUC.trad_conv_eps = 5e-3  # NOTE this is for single panel model
                RUC.max_calls = 300
                RUC.verbose = False
                if args.verbose:
                    if rank == 0:  # only show refinement stats for rank 0
                        RUC.verbose = True
                RUC.run()
                if RUC.hit_break_to_use_curvatures:
                    RUC.num_positive_curvatures = 0
                    RUC.use_curvatures = True
                    RUC.run(setup=False)
            except AssertionError as err:
                print("Rank %d, filename %s Hit assertion error during refinement: %s" % (rank, data.filename, err))
                continue

            angle, ax = RUC.get_correction_misset(as_axis_angle_deg=True)
            if args.startwithtruth:
                C = Crystal(a_tru, b_tru, c_tru, "P43212")
            C.rotate_around_origin(ax, angle)
            C.set_B(RUC.get_refined_Bmatrix())
            a_ref, _, c_ref, _, _, _ = C.get_unit_cell().parameters()
            # compute missorientation with ground truth model
            try:
                angular_offset = compare_with_ground_truth(a_tru, b_tru, c_tru, [C], symbol="P43212")[0]
                print("Rank %d, filename=%s, ang=%f, init_ang=%f, a=%f, init_a=%f, c=%f, init_c=%f" % (
                    rank, data.filename, angular_offset, angular_offset_init, a_ref, a_init, c_ref, c_init))
            except Exception as err:
                print("Rank %d, filename=%s, error %s" % (rank, data.filename, err))

            A_ref = C.get_A()
            ff = os.path.basename(npz_path)
            np.save("crystals/%s_refined" % ff, A_ref)
            # free the memory from diffBragg instance
            RUC.S.D.free_all()
            del img  # not sure if needed here..
            del img_in_photons

            if args.testmode:
                exit()

            # peak at the memory usage of this rank
            mem = getrusage(RUSAGE_SELF).ru_maxrss  # peak mem usage in KB
            mem = mem / 1e6  # convert to GB

            if rank == 0:
                print "RANK 0: %.2g total pixels in %d/%d bboxes (file %d / %d); MemUsg=%2.2g GB" \
                      % (Ntot, len(bboxes), n_bboxes_total,  img_num+1, len(my_shots), mem)

            # TODO: accumulate all pixels
            #self.all_bbox_pixels += data_boxes

        for h in my_open_files.values():
            h.close() 
      
        print ("Rank %d; all subimages loaded!" % rank) 


B = BigData()
fnames = glob(args.glob)
assert(fnames)
B.fnames = fnames
B.load()

