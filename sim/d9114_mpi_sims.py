#!/usr/bin/env libtbx.python

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    has_mpi = True
except ImportError:
    has_mpi = False
    size = 1
    rank = 0

import argparse

parser = argparse.ArgumentParser("make dadat")
parser.add_argument("--savenoiseless", action="store_true")
parser.add_argument("-o", dest='ofile', type=str, help="file out")
parser.add_argument("-odir", dest='odir', type=str, help="file outdir", default="_sims64res")
parser.add_argument("--seed", type=int, dest='seed', default=None, help='random seed for orientation')
parser.add_argument("--sad", action="store_true")
parser.add_argument("--gpu", dest='gpu', action='store_true', help='sim with GPU')
parser.add_argument("--rank-seed", dest='use_rank_as_seed', action='store_true',
                    help="seed the random number generator with worker Id")
parser.add_argument("--masterscale", type=float, default=None)
parser.add_argument("--readoutnoise", action="store_true")
parser.add_argument("--add-bg", dest="add_bg", action='store_true', help="add background")
parser.add_argument("--add-noise", dest="add_noise", action='store_true', help="add noise")
parser.add_argument("--masterscalejitter", type=float, default=0, help="sigma of the master scale")
parser.add_argument("--Ncellsjitter", type=float, default=0, help="sigma of the Ncells")
parser.add_argument("--ucelljitter", type=float, default=0, help="sigma of the trtragonal unit cell constants a and c")
parser.add_argument("--profile", dest="profile", type=str, default=None,
                    choices=["gauss", "round", "square", "tophat"], help="shape of spots determined with this")
parser.add_argument("--cspad", action="store_true")
parser.add_argument("--startidx", type=int, default=0, help="which index in the mastter flux array to begin")
parser.add_argument("--make-background", dest='make_bg', action='store_true',
                    help="Just make the background image and quit")
parser.add_argument("--bg-name", dest='bg_name', default=None,
                    type=str, help="name of the background file, either to make/overwrite, or load (default)")
parser.add_argument("-g", dest='ngpu_per_node', type=int, default=1, help='number of gpu')
parser.add_argument("--overwrite", dest='overwrite', action='store_true', help='overwrite files')
parser.add_argument("--write-img", dest='write_img', action='store_true')
parser.add_argument('-N', dest='nodes', type=int, default=[1, 0], nargs=2, help="Number of nodes, and node id")
parser.add_argument("-trials", dest='num_trials', help='trials per worker',
                    type=int, default=1)
parser.add_argument("--optimize-oversample", action='store_true')
parser.add_argument("--show-params", action='store_true')
parser.add_argument("--kernelspergpu", default=1, type=int, help="how many processes  accessing each gpu")
parser.add_argument("--oversample", type=int, default=0)
parser.add_argument("--Ncells", type=float, default=15)
parser.add_argument("--xtal_size_mm", type=float, default=None)
parser.add_argument("--mos_spread_deg", type=float, default="0.01")
parser.add_argument("--mos_doms", type=int, default=100)
parser.add_argument("--xtal_size_jitter", type=float, default=None)

args = parser.parse_args()

if rank==0:
    print (args)

profile = args.profile
cuda = args.gpu
add_background = args.add_bg
add_noise = args.add_noise
num_nodes, node_id = args.nodes
make_background = args.make_bg
overwrite = args.overwrite
ngpu_per_node = args.ngpu_per_node
ofile = args.ofile
div_tup = (0, 0, 0)  # (0.13, .13, 0.06)  # horiz, verti, stpsz (mrads)

kernels_per_gpu = args.kernelspergpu
smi_stride = 5
GAIN = 28
beam_diam_mm = 1. * 1e-3
exposure_s = 50e-15  # 1 #50e-15  # 50 femtoseconds
mos_spread = args.mos_spread_deg
mos_doms = args.mos_doms
adc_offset = 0

device_Id = rank % ngpu_per_node

import os
import sys
import h5py
import numpy as np

from simtbx.nanoBragg import shapetype, nanoBragg
from scitbx.array_family import flex
from dxtbx.model.crystal import CrystalFactory

from cxid9114.sf import struct_fact_special
from cxid9114 import parameters
from cxid9114.utils import random_rotation
from cxid9114.sim import sim_utils
# from cxid9114.refine.jitter_refine import make_param_list
from cxid9114.geom.single_panel import DET, BEAM
if args.cspad:  # use a cspad for simulation
    from cxid9114.geom.multi_panel import CSPAD
    # from dxtbx.model import Panel
    # put this new CSPAD in the same plane as the single panel detector (originZ)
    for pid in range(len(CSPAD)):
        CSPAD[pid].set_trusted_range(DET[0].get_trusted_range())
        # node_d = CSPAD[pid].to_dict()
        # node_d["origin"] = node_d["origin"][0], node_d["origin"][1], DET[0].get_origin()[2]
        # CSPAD[pid] = Panel.from_dict(node_d)
    # from IPython import embed
    # embed()
    DET = CSPAD  # rename

if args.use_rank_as_seed:
    np.random.seed(rank)
else:
    np.random.seed(args.seed)

sim_path = os.path.dirname(sim_utils.__file__)
spectra_file = os.path.join(sim_path, "../spec/realspec.h5")
sfall_file = os.path.join(sim_path, "../sf/realspec_sfall.h5")

data_fluxes_all = h5py.File(spectra_file, "r")["hist_spec"][()] / exposure_s
#data_fluxes_all = data_fluxes_all[args.startidx:]
ave_flux_across_exp = np.mean(data_fluxes_all, axis=0).sum()
data_sf, data_energies = struct_fact_special.load_sfall(sfall_file)
from cxid9114.parameters import ENERGY_LOW, WAVELEN_LOW
if args.sad:
    print("Rank %d: Loading 4bs7 structure factors!" % rank)
    #_i = np.argmin(np.abs(data_energies - ENERGY_LOW))
    #data_sf = [data_sf[_i]]
    # NOTE there is a bug in the load_4bs7_sf code

    data_sf = struct_fact_special.load_p9()
    #data_sf = struct_fact_special.load_4bs7_sf()
    data_sf = [data_sf]
    

if args.sad:
    #data_energies = np.array([ENERGY_LOW])
    data_energies = np.array([12660.5])

BEAM.set_wavelength(0.9793)

beamsize_mm = np.sqrt(np.pi * (beam_diam_mm / 2) ** 2)

data_fluxes_worker = np.array_split(data_fluxes_all, size)[rank]
data_fluxes_idx = np.array_split(np.arange(data_fluxes_all.shape[0]), size)[rank]
a, b, c, _, _, _ = data_sf[0].unit_cell().parameters()
hall = data_sf[0].space_group_info().type().hall_symbol()
from IPython import embed
embed()

# Each rank (worker)  gets its own output directory
odir = args.odir
odirj = os.path.join(odir, "job%d" % rank)
if not os.path.exists(odirj):
    os.makedirs(odirj)

if add_background:
    assert args.bg_name is not None
    background = h5py.File(args.bg_name, "r")['bigsim_d9114'][()]
    if args.cspad:
        assert background.shape == (64, 185, 194)

print("Rank %d Begin" % rank)
for i_data in range(args.num_trials):
    flux_id = data_fluxes_idx[i_data]
    h5name = "%s_rank%d_data%d_fluence%d.h5" % (ofile, rank, i_data, flux_id)
    h5name = os.path.join(odirj, h5name)
    if os.path.exists(h5name) and not args.overwrite:
        print("Rank %d: skipping- image %s already exists!" \
              % (rank, h5name))
        continue

    print("<><><><><><><")
    print("Rank %d:  trial  %d / %d" % (rank, i_data + 1, args.num_trials))
    print("<><><><><><><")

    # If im the zeroth worker I want to show usage statisitcs:
    if rank == 0 and i_data % smi_stride == 0 and cuda:
        print("GPU status")
        os.system("nvidia-smi")

        print("\n\n")
        print("CPU memory usage")
        mem_usg = """ps -U dermen --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB consumed by CPU user"}'"""
        os.system(mem_usg)

    data_fluxes = data_fluxes_worker[i_data]

    if args.sad:
        data_fluxes = np.array([ave_flux_across_exp])

    a = np.random.normal(a, args.ucelljitter)
    c = np.random.normal(c, args.ucelljitter)
    cryst_descr = {'__id__': 'crystal',  # its al,be,ga = 90,90,90
                   'real_space_a': (a, 0, 0),
                   'real_space_b': (0, a, 0),
                   'real_space_c': (0, 0, c),
                   'space_group_hall_symbol': hall}
    crystal = CrystalFactory.from_dict(cryst_descr)
    # generate a random scale factor within two orders of magnitude to apply to the image data..
    # rotate the crystal using a known rotation
    # crystal = CrystalFactory.from_dict(cryst_descr)

    randnums = np.random.random(3)
    Rrand = random_rotation(1, randnums)
    crystal.set_U(Rrand.ravel())

    ## NOTE start debug hack
    # _A = np.array([-0.00659296, -0.00848504, -0.01388453,
    #               -0.00458244,  0.00930397, -0.01505553,
    #               0.00979487, -0.00135853, -0.01638933])
    # crystal.set_A(_A)
    ## NOTE end debug hack

    # NOTE: This can be used to simulate jitter in the
    # crystal model, but for now we disable jitter
    # params_lst = make_param_list(crystal, DET, BEAM,
    #    1, rot=0.08, cell=.0000001, eq=(1,1,0),
    #    min_Ncell=23, max_Ncell=24,
    #    min_mos_spread=0.02,
    #    max_mos_spread=0.08)
    # Ctruth = params_lst[0]['crystal']
    Ncells = int(np.random.normal(args.Ncells, args.Ncellsjitter))
    Ncells_abc = (Ncells, Ncells, Ncells)

    # the following will override some parameters
    # to aid simulation of a background image using same pipeline
    if make_background:
        print("Rank %d: MAKING BACKGROUND : just at two colors" % rank)
        data_fluxes = [ave_flux_across_exp * .5, ave_flux_across_exp * .5]
        data_energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # should be 8944 and 9034
        data_sf = [1, 1]  # dont care about structure factors when simulating background water scatter
        Ncells_abc = 10, 10, 10
        only_water = True
    else:
        only_water = False

    xtal_size_mm = args.xtal_size_mm
    if args.xtal_size_jitter is not None:
        xtal_size_mm = np.random.normal(xtal_size_mm, args.xtal_size_jitter)

    sim_args = [crystal, DET, BEAM, data_sf, data_energies, data_fluxes]
    masterscale = args.masterscale
    if masterscale is not None:
        masterscale = np.random.normal(masterscale, args.masterscalejitter)
    if masterscale < 0.1:  # FIXME: make me more smart
        master_scale = 0.1
    sim_kwargs = {'pids': None,
                  'profile': args.profile,
                  'cuda': cuda,
                  'oversample': args.oversample,
                  'Ncells_abc': Ncells_abc,
                  'accumulate': True,
                  'mos_dom': mos_doms,
                  'mos_spread': mos_spread,
                  'exposure_s': exposure_s,
                  'beamsize_mm': beamsize_mm,
                  'only_water': only_water,
                  'device_Id': device_Id,
                  'div_tup': div_tup,
                  'master_scale': masterscale,
                  'gimmie_Patt': True,
                  'adc_offset': adc_offset,
                  'show_params': args.show_params,
                  'crystal_size_mm': xtal_size_mm,
                  'one_sf_array': True} #data_sf[0] is not None and data_sf[1] is None}

    print ("Rank %d: SIULATING DATA IMAGE" % rank)
    if args.optimize_oversample:
        oversample = 1
        reference = None
        while 1:
            sim_kwargs['oversample'] = oversample
            simsDataSum, PattFact = sim_utils.sim_colors(*sim_args, **sim_kwargs)
            simsDataSum = np.array(simsDataSum)

            if oversample == 1:
                reference = simsDataSum

            else:
                residual = np.abs(reference - simsDataSum)

                where_one_photon = np.where(simsDataSum > 1)
                N_over_sigma = (residual[where_one_photon] > np.sqrt(reference[where_one_photon])).sum()
                print ("Rank%d : Oversample = %d; |Residual| summed = %f; N over sigma: %d" \
                       % (rank, oversample, residual.sum(), N_over_sigma))
                if np.allclose(simsDataSum, reference, atol=4):
                    print ("Rank %d: Optimal oversample for current parameters: %d\nGoodbyw" % (rank, oversample))
                    sys.exit()
                reference = simsDataSum
            oversample += 1

    else:
        simsDataSum, PattFact = sim_utils.sim_colors(*sim_args, **sim_kwargs)
        #
        spot_scale = PattFact.spot_scale
        simsDataSum = np.array(simsDataSum)
        if args.cspad:
            assert simsDataSum.shape == (64, 185, 194)

    if make_background:
        if args.cspad:
            bg_out = h5py.File(args.bg_name, "w")
            bg_out.create_dataset("bigsim_d9114", data=simsDataSum)
            np.savez("testbg.npz", img=simsDataSum,
                     det=DET.to_dict(),
                     beam=BEAM.to_dict())
        else:
            bg_out = h5py.File(args.bg_name, "w")
            bg_out.create_dataset("bigsim_d9114", data=simsDataSum[0])
            np.savez("testbg_mono.npz", img=simsDataSum[0],
                     det=DET.to_dict(),
                     beam=BEAM.to_dict())
        print ("Rank %d: Background made! Saved to file %s" % (rank, args.bg_name))
        # force an exit here if making a background...
        sys.exit()

    if add_background:
        print("Rank %d: ADDING BG" % rank)
        # background was made using average flux over all shots, so scale it up/down here
        bg_scale = data_fluxes.sum() / ave_flux_across_exp
        # TODO consider varying the background level to simultate jet thickness jitter
        if args.cspad:
            simsDataSum += background * bg_scale
        else:
            simsDataSum[0] += background * bg_scale

    if add_noise:
        print("Rank %d: ADDING NOISE" % rank)
        if args.savenoiseless:
            np.savez(h5name + ".noiseless.npz",
                     img=simsDataSum.astype(np.float32),
                     det=DET.to_dict(), beam=BEAM.to_dict())
        for pidx in range(len(DET)):
            SIM = nanoBragg(detector=DET, beam=BEAM, panel_id=pidx)
            SIM.beamsize_mm = beamsize_mm
            SIM.exposure_s = exposure_s
            SIM.flux = np.sum(data_fluxes)
            SIM.adc_offset_adu = adc_offset
            # SIM.detector_psf_kernel_radius_pixels = 5
            # SIM.detector_psf_type = shapetype.Unknown  # for CSPAD
            SIM.detector_psf_fwhm_mm = 0
            SIM.quantum_gain = GAIN
            if not args.readoutnoise:
                SIM.readout_noise_adu = 0
            SIM.raw_pixels = flex.double(simsDataSum[pidx].ravel())
            SIM.add_noise()
            simsDataSum[pidx] = SIM.raw_pixels.as_numpy_array() \
                .reshape(simsDataSum[pidx].shape)
            SIM.free_all()
            del SIM

    # all_rots =[]
    # from scitbx.matrix import sqr
    # for i_U, U in enumerate(Umats):
    #    Utruth = crystal.get_U()
    #    crystal2 = CrystalFactory.from_dict(cryst_descr)
    #    crystal2.set_U( sqr(U)*sqr(Utruth) )
    #    out = difference_rotation_matrix_axis_angle(crystal, crystal2)
    #    all_rots.append(out[2])
    # assert( np.mean(all_rots) < 1e-7)

    if args.write_img:

        print "Rank %d: SAVING DATAFILE" % rank
        if args.cspad:
            np.savez(h5name + ".npz",
                     img=simsDataSum.astype(np.float32),
                     det=DET.to_dict(), beam=BEAM.to_dict())

        else:
            np.savez(h5name + ".npz",
                     img=simsDataSum[0].astype(np.float32),
                     det=DET.to_dict(), beam=BEAM.to_dict())

        fout = h5py.File(h5name, "w")
        # fout.create_dataset("bigsim_d9114", data=simsDataSum[0].astype(np.float32), compression='lzf')
        fout.create_dataset("crystalA", data=crystal.get_A())
        fout.create_dataset("crystalU", data=crystal.get_U())
        fout.create_dataset("spectrum", data=data_fluxes)
        fout.create_dataset("mos_doms", data=mos_doms)
        fout.create_dataset("mos_spread", data=mos_spread)
        fout.create_dataset("Ncells_abc", data=Ncells_abc)
        fout.create_dataset("divergence_tuple", data=div_tup)
        fout.create_dataset("beamsize_mm", data=beamsize_mm)
        fout.create_dataset("exposure_s", data=exposure_s)
        fout.create_dataset("profile", data=profile)
        fout.create_dataset("xtal_size_mm", data=xtal_size_mm)
        fout.create_dataset("spot_scale", data=spot_scale)
        fout.create_dataset("gain", data=GAIN)
        fout.close()

    print("Rank %d: DonDonee" % rank)
