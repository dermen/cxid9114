#!/usr/bin/env libtbx.python

import argparse
parser = argparse.ArgumentParser("make dadat")
parser.add_argument("-o", dest='ofile', type=str, help="file out")
parser.add_argument("-odir", dest='odir', type=str, help="file outdir", default="_sims64res")
parser.add_argument("--seed",type=int, dest='seed', default=None, help='random seed for orientation' )
parser.add_argument("--gpu", dest='gpu', action='store_true', help='sim with GPU')
parser.add_argument("--add-bg", dest="add_bg",action='store_true',help="add background" )
parser.add_argument("--add-noise", dest="add_noise",action='store_true',help="add noise" )
parser.add_argument("--profile", dest="profile", type=str, default=None,
    choices=["gauss", "round", "square", "tophat"],help="overrides --gauss")
parser.add_argument("--gauss", dest="Gauss", action='store_true', help="use gaussian profile")
parser.add_argument("--make-background", dest='make_bg', action='store_true',
    help="Just make the background image and quit")
parser.add_argument("--bg-name", dest='bg_name', default='background64.h5',
    type=str, help="name of the background file, either to make/overwrite, or load (default)")
parser.add_argument("-g", dest='ngpu', type=int, default=1,help='number of gpu' )
parser.add_argument("--overwrite", dest='overwrite',action='store_true',help='overwrite files' )
parser.add_argument("--write-img", dest='write_img', action='store_true')
parser.add_argument('-N', dest='nodes', type=int, default=[1,0], nargs=2, help="Number of nodes, and node id")
parser.add_argument("-trials", dest='num_trials', help='trials per worker', 
    type=int, default=1 )
args = parser.parse_args()


Gauss = args.Gauss
cuda = args.gpu
add_background = args.add_bg
add_noise = args.add_noise
num_nodes, node_id = args.nodes
make_background = args.make_bg
bg_name = args.bg_name
overwrite = args.overwrite
ngpu = args.ngpu
ofile = args.ofile
div_tup = (0,0,0) #(0.13, .13, 0.06)  # horiz, verti, stpsz (mrads)

kernels_per_gpu = 1
smi_stride = 5
GAIN=28
beam_diam_mm = 1.*1e-3
exposure_s = 50e-15 #1 #50e-15  # 50 femtoseconds

mos_spread = .025
oversample = 0

Deff_A = 3000  # 300 nm domains  inside crystal
crystal_length_um = 2 # 2. micron crystal (length) 


def main(rank):

    device_Id = rank % ngpu
    worker_Id = node_id*ngpu + rank  # support for running on multiple N-node GPUs to increase worker pool

    import os
    import sys
    from cxid9114.sf import struct_fact_special

    import h5py
    import numpy as np
   
    from simtbx.nanoBragg import shapetype, nanoBragg 
    from scitbx.array_family import flex
    from dxtbx.model.crystal import CrystalFactory

    from cxid9114 import parameters
    from cxid9114.utils import random_rotation
    from cxid9114.sim import sim_utils
    #from cxid9114.refine.jitter_refine import make_param_list
    from cxid9114.geom.single_panel import DET, BEAM
    
    np.random.seed(args.seed)
    
    crystal_length_um = np.random.normal(2, 0.5) # 2. micron crystal (length) 
    
    sim_path = os.path.dirname(sim_utils.__file__)
    spectra_file = os.path.join( sim_path, "../spec/realspec.h5")
    sfall_file = os.path.join(sim_path, "../sf/realspec_sfall.h5")

    data_fluxes_all = h5py.File(spectra_file, "r")["hist_spec"][()] / exposure_s
    ave_flux_across_exp = np.mean(data_fluxes_all,axis=0).sum()
    data_sf, data_energies = struct_fact_special.load_sfall(sfall_file)
    
    beamsize_mm = np.sqrt(np.pi*(beam_diam_mm/2)**2)
    
    # TODO: verify this works for all variants of parallelization (multi 8-GPU nodes, multi kernels per GPU etc)
    data_fluxes_worker = np.array_split(data_fluxes_all, ngpu * kernels_per_gpu )[worker_Id]

    # use micrys to determine size and number of domains in crystal
    micrys = sim_utils.microcrystal(Deff_A = Deff_A, length_um = crystal_length_um, 
        beam_diameter_um = beam_diam_mm*1000)
    Ntotal = micrys.number_of_cells(data_sf[0].unit_cell())
    Ncells_abc = (Ntotal,Ntotal,Ntotal)  
    total_doms = int(micrys.domains_per_crystal)
    
    intensity_boost = total_doms * (beam_diam_mm*1e3 * np.pi / crystal_length_um / 4)

    print("%.2f total doms; %.2f intensity boost" % (total_doms, intensity_boost))
    mos_doms = 250
    
    a,b,c,_,_,_ = data_sf[0].unit_cell().parameters()
    hall = data_sf[0].space_group_info().type().hall_symbol()
    cryst_descr = {'__id__': 'crystal',  # its al,be,ga = 90,90,90
                  'real_space_a': (a, 0, 0),
                  'real_space_b': (0, b, 0),
                  'real_space_c': (0, 0, c),
                  'space_group_hall_symbol': hall}
    
    # Each rank (worker)  gets its own output directory
    odir = args.odir
    odirj = os.path.join(odir, "job%d" % worker_Id)
    if not os.path.exists(odirj):
        os.makedirs(odirj)

    crystal = CrystalFactory.from_dict(cryst_descr)
    print("Rank %d Begin" % worker_Id)
    for i_data in range( args.num_trials):

        print("<><><><><><><")
        print("Job %d:  trial  %d / %d" % ( worker_Id, i_data+1, args.num_trials ))
        print("<><><><><><><")
        
        # If im the zeroth worker I want to show usage statisitcs: 
        if worker_Id == 0 and i_data % smi_stride == 0 and cuda:
            print("GPU status")
            os.system("nvidia-smi")

            print("\n\n")
            print("CPU memory usage")
            mem_usg= """ps -U dermen --no-headers -o rss | awk '{ sum+=$1} END {print int(sum/1024) "MB consumed by CPU user"}'"""
            os.system(mem_usg)

        data_fluxes = data_fluxes_worker[0]

        # generate a random scale factor within two orders of magnitude to apply to the image data.. 

        # rotate the crystal using a known rotation
        #crystal = CrystalFactory.from_dict(cryst_descr)

        randnums = np.random.random(3)
        Rrand = random_rotation(1, randnums)
        crystal.set_U(Rrand.ravel())

        # NOTE: This can be used to simulate jitter in the 
        # crystal model, but for now we disable jitter
        #params_lst = make_param_list(crystal, DET, BEAM, 
        #    1, rot=0.08, cell=.0000001, eq=(1,1,0),
        #    min_Ncell=23, max_Ncell=24, 
        #    min_mos_spread=0.02, 
        #    max_mos_spread=0.08)
        #Ctruth = params_lst[0]['crystal']
        
        
        # the following will override some parameters
        # to aid simulation of a background image using same pipeline
        if make_background:
            print("MAKING BACKGROUND : just at two colors")
            data_fluxes = [ ave_flux_across_exp*.5, ave_flux_across_exp*.5]
            data_energies = [parameters.ENERGY_LOW, parameters.ENERGY_HIGH]  # should be 8944 and 9034
            data_sf = [1,1]  # dont care about structure factors when simulating background water scatter
            Ncells_abc = (10,10,10)
            only_water=True
        else:
            only_water=False

        print ("SIULATING DATA IMAGE")
        simsDataSum, PattFact = sim_utils.sim_colors(
            crystal, DET, BEAM, data_sf, 
            data_energies, 
            data_fluxes,
            pids=None, Gauss=Gauss, cuda=cuda,
            oversample=oversample,
            Ncells_abc=Ncells_abc, accumulate=True, mos_dom=mos_doms, 
            mos_spread=mos_spread, boost=intensity_boost,
            exposure_s=exposure_s, beamsize_mm=beamsize_mm,
            only_water=only_water, device_Id=device_Id, div_tup=div_tup, gimmie_Patt=True)
            
        Umats = PattFact.Umats
        simsDataSum = np.array(simsDataSum)
        
        if make_background:
            bg_out = h5py.File(bg_name, "w")
            bg_out.create_dataset("bigsim_d9114",data=simsDataSum[0])
            print ("Background made! Saved to file %s" % bg_name)
            # force an exit here if making a background...
            sys.exit()
        
        if add_background:
            print("ADDING BG")
            background = h5py.File(bg_name, "r")['bigsim_d9114'][()]
            # background was made using average flux over all shots, so scale it up/down here
            bg_scale = data_fluxes.sum() / ave_flux_across_exp
            # TODO consider varying the background level to simultate jet thickness jitter
            simsDataSum[0] += background * bg_scale

        if add_noise:
            print("ADDING NOISE")
            for pidx in range(len(DET)):  
                SIM = nanoBragg(detector=DET, beam=BEAM, panel_id=pidx)
                SIM.beamsize_mm = beamsize_mm
                SIM.exposure_s = exposure_s
                SIM.flux = np.sum(data_fluxes)
                SIM.detector_psf_kernel_radius_pixels=5;
                SIM.detector_psf_type=shapetype.Unknown  # for CSPAD
                SIM.detector_psf_fwhm_mm = 0
                SIM.quantum_gain = GAIN
                SIM.raw_pixels = flex.double(simsDataSum[pidx].ravel())
                SIM.add_noise()
                simsDataSum[pidx] = SIM.raw_pixels.as_numpy_array()\
                    .reshape(simsDataSum[0].shape)    
                SIM.free_all()
                del SIM

    
        if args.write_img:
            print "SAVING DATAFILE"
            h5name = "%s_rank%d_data%d.h5" % (ofile, worker_Id, i_data)
            h5name = os.path.join( odirj, h5name)
            fout = h5py.File(h5name,"w" ) 
            fout.create_dataset("bigsim_d9114", data=simsDataSum[0])
            fout.create_dataset("crystalA", data=crystal.get_A() )
            fout.create_dataset("crystalU", data=crystal.get_U() )
            fout.create_dataset("spectrum", data=data_fluxes)
            fout.create_dataset("intensity_boost", data=intensity_boost)
            fout.create_dataset("Deff_A", data=Deff_A)
            fout.create_dataset("mos_doms", data=mos_doms)
            fout.create_dataset("mos_spread", data=mos_spread)
            fout.create_dataset("Ncells_abc", data=Ncells_abc)
            fout.create_dataset("divergence_tuple", data=div_tup)
            fout.create_dataset("beamsize_mm", data=beamsize_mm)
            fout.create_dataset("exposure_s", data=exposure_s)
            fout.create_dataset("Umats", data=Umats)
            # TODO: write out the Umats
            # TODO: write out all other parameters
            fout.close()  

        print("DonDonee")

if __name__=="__main__":
    from joblib import Parallel,delayed
    Parallel(n_jobs=ngpu*kernels_per_gpu)(\
        delayed(main)(rank) for rank in range(ngpu*kernels_per_gpu) )


