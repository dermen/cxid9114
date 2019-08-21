

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# import functions on rank 0 only
if rank==0:
    from h5py import File as h5py_File
    from cxid9114.integrate.integrate_utils import Integrator
    from numpy import array, sqrt, percentile
    from numpy import zeros as np_zeros
    from numpy import sum as np_sum
    from psutil import Process
    from os import getpid
else:
    h5py_File = None
    Integrator = None
    array = sqrt = percentile = np_zeros = np_sum = None
    Process = None
    getpid = None

h5py_File = comm.bcast(h5py_File, root=0)
Integrator = comm.bcast(Integrator, root=0)
array = comm.bcast(array, root=0)
sqrt = comm.bcast(sqrt, root=0)
percentile = comm.bcast(percentile, root=0)
np_zeros = comm.bcast(np_zeros, root=0)
np_sum = comm.bcast(np_sum, root=0)
Process = comm.bcast(Process, root=0)
getpid = comm.bcast( getpid, root=0)

def get_memory_usage():
    """Return the memory usage in Mo."""
    process = Process(getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

class BigData:

    def __init__(self):
        self._load()

    #@profile
    def _load(self):

        # some parameters
        int_radius = 5
        gain = 28
        Nload = None
        # bbox data is stored in 39 h5py_Files
        Nfiles = 2 #39
        fname_template = "/global/project/projectdirs/lcls/dermen/process_rank%d.h5"
        fnames = [fname_template % i_f for i_f in range( Nfiles)]

        # NOTE: for reference, inside each h5 file there is 
        #   [u'Amatrices', u'Hi', u'bboxes', u'h5_path']
        
        # get the total number of shots using worker 0
        if rank==0:
            print("I am root. I am calculating total number of shots")
            h5s = [h5py_File(f, "r") for f in fnames]
            Nshots_per_file = [ h["h5_path"].shape[0] for h in h5s]
            Nshots_tot = sum(Nshots_per_file)
            print("I am root. Total number of shots is %d" % Nshots_tot)

            print("I am root. I will divide shots amongst workers.")
            shot_tuples = []
            for i_f, fname in enumerate( fnames):
                fidx_shotidx = [(i_f, i_shot) for i_shot in range(Nshots_per_file[i_f])] 
                shot_tuples += fidx_shotidx

            from numpy import array_split
            print ("I am root. Number of uniques = %d" % len(set(shot_tuples)))
            shots_for_rank = array_split( shot_tuples, size)

            # close the open h5s.. 
            for h in h5s:
                h.close()

        else:
            Nshots_tot = None
            shots_for_rank = None
            h5s = None
        
        #Nshots_tot = comm.bcast( Nshots_tot, root=0)
        shots_for_rank = comm.bcast( shots_for_rank, root=0)
        #h5s = comm.bcast( h5s, root=0)  # pull in the open hdf5 files
        
        my_shots =  shots_for_rank[rank]
        if Nload is not None:
            my_shots =  my_shots[:Nload]

        # open the unique filenames for this rank
        # TODO: check max allowed pointers to open hdf5 file
        my_unique_fids = set([fidx for fidx,_ in my_shots])
        my_open_files = {fidx: h5py_File( fnames[fidx], "r") for fidx in my_unique_fids}
        
        Ntot = 0
        self.all_bbox_pixels = []
        for img_num, ( fname_idx, shot_idx) in  enumerate( my_shots):

            h = my_open_files[fname_idx]
            
            # load the dxtbx image data directly:
            img_path = h["h5_path"][shot_idx]
            img_h5 = h5py_File(img_path, "r")
            img_data = img_h5["bigsim_d9114"][()]  # LZF decompression, but not a bottleneck

            bbox_dset = h["bboxes"]["shot%d" % shot_idx]
            n_bboxes = bbox_dset.shape[0] 
            is_a_keeper = h["bboxes"]["keepers%d" % shot_idx][()]
            
            bboxes = [bbox_dset[i_bb] for i_bb in range(n_bboxes) if is_a_keeper[i_bb]]
          
            # actually load the pixels...  
            data_boxes = [ img_data[j1:j2, i1:i2] for i1,i2,j1,j2 in bboxes]

            tot_pix = [ (j2-j1)*(i2-i1) for i1,i2,j1,j2 in bboxes]
            Ntot += sum(tot_pix)

            print "%g total pixels in %d / %d bboxes..  (file %d / %d);" % (Ntot, len(bboxes), n_bboxes,  img_num+1, len(my_shots))
            assert len(bboxes) / float( n_bboxes) < 0.15  # sanity check
            self.all_bbox_pixels += data_boxes
       
        for h in my_open_files.values():
            h.close() 
      
        print ("Rank %d; all subimages loaded!" % rank) 

        #all_bbox_pixels = comm.gather( all_bbox_pixels, root=0)
        #
        #if rank==0:
        #    all_bbox_pixels = [pixel_img for lst in all_bbox_pixels  for pixel_img in lst]
        #    Ntot_pix = sum([ pixel_img.size for pixel_img in all_bbox_pixels])
        #    print
        #    print("<><><><><><><<><><><><><><><><><><><><><><>")
        #    print("I am root. total bboxes=%d, Total pixels=%g" % (len(all_bbox_pixels), Ntot_pix))
        #    print("<><><><><><><<><><><><><><><><><><><><><><>")
        #    print 
       

if __name__=="__main__" :
    BigData()
