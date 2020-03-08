#!/usr/bin/env libtbx.python

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    has_mpi = True
    print("has mpi")
except ImportError:
    print("NOOOOOOOOOOOO has mpi")
    exit()
    has_mpi = False
    rank = 0
    size = 1

from IPython import embed

if rank == 0:
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser("options to filter bboxes on each shot",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plot", default=None, type=float)
    parser.add_argument("--p9", action="store_true")
    parser.add_argument("--reshigh", type=float, default=2.5, help="high res limit for selecting bboxes")
    parser.add_argument("--reslow", type=float, default=3.5, help="low res limit for selecting bboxes")
    parser.add_argument("--tiltfilt", default=None, type=float, help="minimum rms for the tilt plane fit")
    parser.add_argument("--tilterrmax", default=None, type=float, help="minimum bg plane parameter variance")
    parser.add_argument("--onboundary", action="store_true", help="include spots that are close to the panel boundary")
    parser.add_argument("--notindexed", action="store_true", help="include spots that were flagged as not indexed")
    parser.add_argument("--snrmin", type=float, default=None, help="minimum SNR for selecting bboxes")
    parser.add_argument("--gain", type=float, default=28)
    parser.add_argument("--glob", type=str, required=True, help="glob for selecting files (output files of process_mpi")
    parser.add_argument("--keeperstag", type=str, default="keepers", help="name of keepers boolean array")
    args = parser.parse_args()
    print(args)

# import functions on rank 0 only
if rank == 0:
    from h5py import File as h5py_File
    from glob import glob
    from numpy import load as numpy_load
    from cxid9114.integrate.integrate_utils import Integrator
    from numpy import array, sqrt, percentile
    from numpy import zeros as np_zeros
    from numpy import sum as np_sum
    from psutil import Process
    from os import getpid
    import pylab as plt
else:
    h5py_File = None
    numpy_load = None
    glob = None
    Integrator = None
    array = sqrt = percentile = np_zeros = np_sum = None
    Process = None
    getpid = None
    args = 0

if has_mpi:
    glob = comm.bcast(glob, root=0)
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
    args = comm.bcast(args, root=0)

import dxtbx

def get_memory_usage():
    """Return the memory usage in Mo."""
    process = Process(getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


# @profile
def main():
    # some parameters
    int_radius = 5
    gain = args.gain
    # data is stored in 39 h5py_Files
    resmin = args.reshigh  # high res cutoff
    resmax = args.reslow  # low res cutoff
    fnames = glob(args.glob)

    # NOTE: for reference, inside each h5 file there is
    #   [u'Amatrices', u'Hi', u'bboxes', u'h5_path']

    # get the total number of shots using worker 0
    if rank == 0:
        print("I am root. I am calculating total number of shots")
        h5s = [h5py_File(f, "r") for f in fnames]
        Nshots_per_file = [h["h5_path"].shape[0] for h in h5s]
        Nshots_tot = sum(Nshots_per_file)
        print("I am root. Total number of shots is %d" % Nshots_tot)

        print("I am root. I will divide shots amongst workers.")
        shot_tuples = []
        for i_f, fname in enumerate(fnames):
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

    # Nshots_tot = comm.bcast( Nshots_tot, root=0)
    if has_mpi:
        shots_for_rank = comm.bcast(shots_for_rank, root=0)
    # h5s = comm.bcast( h5s, root=0)  # pull in the open hdf5 files

    my_shots = shots_for_rank[rank]

    # open the unique filenames for this rank
    # TODO: check max allowed pointers to open hdf5 file
    my_unique_fids = set([fidx for fidx, _ in my_shots])
    my_open_files = {fidx: h5py_File(fnames[fidx], "r") for fidx in my_unique_fids}

    Ntot = 0
    all_kept_bbox = []
    all_is_kept_flags = []
    for img_num, (fname_idx, shot_idx) in enumerate(my_shots):
        #import numpy as np
        #idx = np.loadtxt("idx_list.txt")
        #f = open("good_alist2.txt", "w")
        #for i in idx:
        #    fi, si= my_shots[int(i)]
        #    h = my_open_files[fi]
        #    img_path = h["h5_path"][si]
        #    print >> f, img_path
        #f.close()
        #exit()

        h = my_open_files[fname_idx]

        # load the dxtbx image data directly:
        # NOTE: h5_path is really the image file path
        img_path = h["h5_path"][shot_idx]

        loader = dxtbx.load(img_path)
        img_data = loader.get_raw_data().as_numpy_array()

        bboxes = h["bboxes"]["shot%d" % shot_idx][()]
        panel_ids = h["panel_ids"]["shot%d" % shot_idx][()]
        nspots = len(bboxes)

        # use the known cell to compute the resolution of the spots
        reso = h["resolution"]["shot%d" % shot_idx][()]

        in_reso_ring = array([resmin < d < resmax for d in reso])

        # Dirty integrater, sets integration region as disk of diameter 2*int_radius pixels
        if len(img_data.shape) == 2:  # single panel image
            assert len(set(panel_ids)) == 1  # sanity check
            img_data = [img_data]

        is_a_keeper = [in_reso_ring[i_spot] for i_spot in range(nspots)]

        hgroups = h.keys()

        if args.snrmin is not None:
            if "SNR_Leslie99" in hgroups:
                SNR = h["SNR_Leslie99"]["shot%d" % shot_idx][()]
            else:
                if rank == 0:
                    print("WARNING USING DIRTY SNR ESTIMATE!")
                dirties = {pid: Integrator(img_data[pid], int_radius=int_radius, gain=gain)
                       for pid in set(panel_ids)}

                int_data = [dirties[pid].integrate_bbox_dirty(bb) for pid, bb in zip(panel_ids, bboxes)]

                # signal, background, variance  # these are from the paper Leslie '99
                s, b, var = map(array, zip(*int_data))
                SNR = s / sqrt(var)
            is_a_keeper = [k and snr >= args.snrmin for k, snr in zip(is_a_keeper, SNR)]

        if "tilt_rms" in hgroups:
            if args.tiltfilt is not None:
                tilt_rms = h["tilt_rms"]["shot%d" % shot_idx][()]
                is_a_keeper = [k and rms < args.tiltfilt for k, rms in zip(is_a_keeper, tilt_rms)]

        if "tilt_error" in hgroups:
            if args.tilterrmax is not None:
                tilt_err = h["tilt_error"]["shot%d" % shot_idx][()]
                is_a_keeper = [k and err <= args.tilterrmax for k, err in zip(is_a_keeper, tilt_err)]
        else:
            if rank == 0:
                print ("WARNING: tilt_error not in hdf5 file")

        if "indexed_flag" in hgroups:
            #TODO change me to assume indexed_flag is a bool
            if not args.notindexed:
                indexed_flag = h["indexed_flag"]["shot%d" % shot_idx][()]
                is_a_keeper = [k and (idx > 0) for k, idx in zip(is_a_keeper, indexed_flag)]
        else:
            if rank == 0:
                print ("WARNING: indexed_flag not in hdf5 file")

        if "is_on_boundary" in hgroups:
            if not args.onboundary:
                on_boundary = h["is_on_boundary"]["shot%d" % shot_idx][()]
                is_a_keeper = [k and not onbound for k, onbound in zip(is_a_keeper, on_boundary)]
        else:
            if rank == 0:
                print ("WARNING: is_on_boundary not in hdf5 file")

        if rank == 0:
            print("Keeping %d out of %d spots" % (sum(is_a_keeper), nspots))

        if rank == 0 and args.plot is not None:
            for pid in set(panel_ids):
                plt.gcf().clear()
                import numpy as np
                m = np.median(img_data[pid])
                s = np.std( img_data[pid][img_data[pid] > 10])
                vmin = m-s
                vmax = m+5*s
                plt.imshow(img_data[pid], vmax=vmax,vmin=vmin, cmap='viridis')
                for i_spot in range(nspots):
                    if not is_a_keeper[i_spot]:
                        continue
                    if not panel_ids[i_spot] == pid:
                        continue
                    x1, x2, y1, y2 = bboxes[i_spot]
                    patch = plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fc='none', ec='r')
                    plt.gca().add_patch(patch)
                plt.title("image %s\nrank%d , index %d, Panel=%d" % (img_path, rank, img_num, pid) )
                if args.plot == -1:
                    plt.show()
                else:
                    plt.draw()
                    plt.pause(args.plot)

        kept_bboxes = [bboxes[i_bb] for i_bb in range(len(bboxes)) if is_a_keeper[i_bb]]

        tot_pix = [(j2 - j1) * (i2 - i1) for i_bb, (i1, i2, j1, j2) in enumerate(kept_bboxes)]
        Ntot += sum(tot_pix)
        if rank == 0:
            print "%g total pixels (file %d / %d)" % (Ntot, img_num + 1, len(my_shots))
        all_kept_bbox += map(list, kept_bboxes)
        all_is_kept_flags += [(fname_idx, shot_idx, is_a_keeper)]  # store this information, write to disk

    # close the open hdf5 files so we can write to them again
    for h in my_open_files.values():
        h.close()

    print("END OF LOOP")
    print "Rank %d; total bboxes=%d; Total pixels=%g" % (rank, len(all_kept_bbox), Ntot)
    all_kept_bbox = MPI.COMM_WORLD.gather(all_kept_bbox, root=0)
    all_is_kept_flags = MPI.COMM_WORLD.gather(all_is_kept_flags, root=0)

    if rank == 0:
        all_kept_bbox = [bbox for bbox_lst in all_kept_bbox for bbox in bbox_lst]
        Ntot_pix = sum([(j2 - j1) * (i2 - i1) for i1, i2, j1, j2 in all_kept_bbox])
        print
        print("<><><><><><><<><><><><><><><><><><><><><><>")
        print("I am root. total bboxes=%d, Total pixels=%g" % (len(all_kept_bbox), Ntot_pix))
        print("<><><><><><><<><><><><><><><><><><><><><><>")
        print

        print("I am root. I will store flags for each bbox on each shot")

        all_flag_info = [i for sl in all_is_kept_flags for i in sl]  # flatten

        # open the hdf5 files in read+write mode and store the bbox keeper flags
        h5s = {i_f: h5py_File(f, "r+") for i_f, f in enumerate(fnames)}

        for i_info, (fidx, shot_idx, keeper_flags) in enumerate(all_flag_info):
            bbox_grp = h5s[fidx]["bboxes"]

            flag_name = "%s%d" % (args.keeperstag, shot_idx)

            if flag_name in bbox_grp:
                del bbox_grp[flag_name]

            bbox_grp.create_dataset(flag_name, data=keeper_flags, dtype=bool, compression='lzf')

            if i_info % 5 == 0:
                print ("I am root. I saved bbox selection flags ( %d / %d ) " % (i_info + 1, len(all_flag_info)))

        # close the open files..
        for h in h5s.values():
            h.close()


if __name__ == "__main__":
    main()
