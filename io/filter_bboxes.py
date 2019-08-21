
from mpi4py import MPI

import h5py
from cxid9114.integrate import integrate_utils

import numpy as np

# 
# size = MPI.COMM_WORLD.size

import psutil
import os

def get_memory_usage():
    """Return the memory usage in Mo."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

@profile
def main():
    rank = MPI.COMM_WORLD.rank

    h = h5py.File("/global/project/projectdirs/lcls/dermen/process_rank%d.h5" % rank, "r")
    paths = h["h5_path"][()]
    bbox_datagrp = h["bboxes"]

    Ntot = 0
    all_tot_pix = []
    
    for img_num, img_path in enumerate(paths[:5]):
        print "rank %d, memory usage = %.3f Mo" % (rank, get_memory_usage())

        img_h5 = h5py.File(img_path, "r")
        #img_data = img_h5["data32"][()]
        img_data = img_h5["bigsim_d9114"][()]
        
        bboxes = bbox_datagrp["shot%d" % img_num][()]

        #data_bbs = [img_data[ j1:j2, i1:i2] for i1,i2,j1,j2 in bboxes]
        I = integrate_utils.Integrator(img_data, int_radius=5, gain=28)
        int_data = [I.integrate_bbox_dirty( bb) for bb in bboxes]

        # signal, background, variance
        s,b,var = map(np.array, zip(*int_data) )
        snr = s / np.sqrt(var)

        # keep the top 10% 
        is_a_keeper =  snr > np.percentile(snr, 90)

        data_boxes = [ img_data[j1:j2, i1:i2] for i_bb,(i1,i2,j1,j2) in enumerate(bboxes) if is_a_keeper[i_bb]]

        tot_pix = [ (j2-j1)*(i2-i1) for i_bb,(i1,i2,j1,j2) in enumerate(bboxes) if is_a_keeper[i_bb]]
        Ntot += sum(tot_pix)

        print "%g total pixels (file %d / %d)" % (Ntot, img_num+1, len(paths))
        all_tot_pix +=  tot_pix
    
    print "Rank %d; total bboxes=%d; Total pixels=%g" % (rank, len(all_tot_pix), Ntot)
    all_tot_pix = MPI.COMM_WORLD.gather( all_tot_pix, root=0)
    
    if rank==0:
        all_tot_pix = [tot_pix for SL in all_tot_pix for tot_pix in SL]
        Ntot_pix = sum([ (j2-j1)*(i2-i1) for i1,i2,j1,j2 in bboxes])
        print
        print("<><><><><><><<><><><><><><><><><><><><><><>")
        print("Rank %d; total bboxes=%d, Total pixels=%g" % (rank, len(all_tot_pix), Ntot_pix))
        print("<><><><><><><<><><><><><><><><><><><><><><>")
        print

if __name__=="__main__" :
    main()
