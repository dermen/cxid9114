
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("input", type=str, help="input concat pandas pickles from hopper run with refine.Fhkl=True")
parser.add_argument("outdir", type=str, help="output folder")
parser.add_argument("--prefix", type=str, help="output file prefix", default="mergy-poo")
parser.add_argument("--thresh", default=5, type=float, help="outlier intensity threshold (amongst ASU equivalents")
parser.add_argument("--nbins", default=15, type=int, help="number of resolution bins for merge stats")
parser.add_argument("--minSNR", default=0, type=float, help="min signal to noise for a spot")
parser.add_argument("--minVar", default=1e-7, type=float, help="min variance for spot to merge")
parser.add_argument("--maxVar", default=1e12, type=float, help="max variance for spot to merge")
parser.add_argument("--minMeas", default=3, type=int, help="minimum multiplicity")
parser.add_argument("--weighted", action="store_true")
parser.add_argument("--median", action="store_true")
parser.add_argument("--halvesOnly", action="store_true")
ARGS = parser.parse_args()

import pandas
import os
from simtbx.diffBragg import utils
import numpy as np
import sys
from itertools import groupby
from libtbx.mpi4py import MPI
import cc_merge

COMM = MPI.COMM_WORLD


def correct_I(hkl, scale, scale_var, Flookup):
    merge_data = []
    for i, h in enumerate(hkl):
        if h not in Flookup:
            continue
        s = scale[i]
        s_var = scale_var[i]
        I_mtz = Flookup[h] ** 2
        I = s * I_mtz
        with np.errstate(over='ignore'):
            I_var = s_var * I_mtz ** 2
        merge_data.append((h, I, I_var))
    return merge_data


def merge_intensities(_I, _varI, thresh=ARGS.thresh, minvar=ARGS.minVar, maxvar=ARGS.maxVar, min_meas=ARGS.minMeas,
                      weighted=ARGS.weighted, use_med=ARGS.median, min_snr=ARGS.minSNR):
    I = _I.copy()
    varI = _varI.copy()
    sel_notnull = (~np.isnan(I)) * (~np.isnan(varI))
    sel_goodvar = (varI >= minvar) * (varI < maxvar)
    sel = sel_notnull * sel_goodvar
    if not np.any(sel):
        return None

    I = I[sel]
    varI = varI[sel]

    inliers = ~utils.is_outlier(I, thresh)
    if not np.any(inliers):
        return None

    I = I[inliers]
    varI = varI[inliers]

    snr = I/np.sqrt(varI)
    above_snr_thresh = snr >= min_snr
    if above_snr_thresh.sum() < min_meas:
        return None
    I = I[above_snr_thresh]
    varI = varI[above_snr_thresh]

    if weighted:
        wt_I = I / varI
        wt = 1/varI
        mergeI = wt_I.sum() / wt.sum()
        errorI = 1 / np.sqrt(wt.sum())
    else:
        if use_med:
            mergeI = np.median(I)
        else:
            mergeI = np.mean(I)
        varI_sum = varI.sum()
        if varI_sum < 0:
            return None

        errorI = np.sqrt(varI_sum) / np.sqrt(len(I))
        if use_med:
            errorI = errorI * 1.253

    return mergeI, errorI, len(I)


def get_overlaps(df):
    df['basename'] = [os.path.basename(f).split("_indexed")[0] for f in df.opt_exp_name]
    gb = df.groupby(["basename"])

    overlaps = []
    for i_g,g in enumerate(gb.groups):
        if i_g % COMM.size != COMM.rank:
            continue

        df_g = gb.get_group(g)
        if len(df_g) > 1:
            #fnames = glob.glob(
            #    "/global/cfs/cdirs/lcls/dermen/d9114_data/hopper_out/22/perl.fhkl_25000-30000/Fhkl_scale/rank*/%s*" % key)
            fnames = []
            for f in df_g.opt_exp_name:
                base_f = f.replace("expers", "Fhkl_scale").split(".expt")[0]
                template_f = base_f + "_channel%d_scale.npz"
                for i_chan in [0,1]:
                    fhkl_f = template_f % i_chan
                    if os.path.exists(fhkl_f):
                        fnames.append(fhkl_f)

            overlap_hkls_chan = []
            nxtal = 0
            ave_num_hkl = 0
            ncount = 0
            for i_chan in [0, 1]:
                fs = []
                for f in fnames:
                    if "channel%d" % i_chan in f:
                        fs.append(f)
                nxtal = max(nxtal, len(fs))
                assert len(fs) == 2 or len(fs) == 3
                hkl_lists = []
                for f in fs:
                    h = list(map(tuple, np.load(f)["asu_hkl"]))
                    ave_num_hkl += len(h)
                    ncount +=1
                    hkl_lists.append(h)
                overlap_hkls = set(hkl_lists[0]).intersection(*hkl_lists[1:])
                overlap_hkls_chan.append(overlap_hkls)

            ave_num_hkl = ave_num_hkl / ncount
            overlap_hkls= overlap_hkls_chan[0].union(overlap_hkls_chan[1])
            if COMM.rank==0:
                nbad = len(overlap_hkls)
                frac_bad = nbad / ave_num_hkl *100.
                print("basename %s has %d xtals and %d (%.2f%%) bad hkls= " % (g, nxtal, nbad, frac_bad), overlap_hkls)
            overlaps.append( [g, overlap_hkls])

    overlaps = COMM.reduce(overlaps)
    if COMM.rank==0:
        overlaps = {g: hkls for g,hkls in overlaps}
    overlaps = COMM.bcast(overlaps)
    return overlaps



def main(df, outname, overlaps):
    mtz_name = "/global/cfs/cdirs/lcls/dermen/d9114_data/merging/merge2/iobs_all.mtz"
    mtz_column = "Iobs,SIGIobs"
    Famp = utils.open_mtz(mtz_name, mtz_column)
    Famp = Famp.expand_to_p1()
    Famp = Famp.generate_bijvoet_mates()

    Flookup = {h: val for h, val in zip(Famp.indices(), Famp.data())}

    merge0, merge1 = [], []
    for i_f, (f, bname) in enumerate(zip(df.opt_exp_name, df.basename)):
        if i_f % COMM.size != COMM.rank:
            continue
        base_f = f.replace("expers", "Fhkl_scale").split(".expt")[0]
        template_f = base_f + "_channel%d_scale.npz"

        overlap_hkls = set()
        if bname in overlaps:
            overlap_hkls = overlaps[bname]

        for i_chan in [0, 1]:
            scale_f = template_f % i_chan
            if not os.path.exists(scale_f):
                continue
            d = np.load(scale_f)
            hkl, scale, scale_var = d['asu_hkl'][()], d['scale_fac'][()], d['scale_var'][()]

            hkl = list(map(tuple, hkl))

            if overlap_hkls:
                keep_hkls = [h not in overlap_hkls for h in hkl]
                hkl = list(map(tuple, np.array(hkl)[keep_hkls]))
                scale = scale[keep_hkls]
                scale_var = scale_var[keep_hkls]

            merge_data = correct_I(hkl, scale, scale_var, Flookup)
            if i_chan == 0:
                merge0 += merge_data
            else:
                merge1 += merge_data
        # if i_f==200#:
        #    break
        if COMM.rank == 0:
            print("Loaded and corrected intensities %d/%d" % (i_f + 1, len(df)), flush=True)

    if COMM.rank == 0:
        print("reducing", flush=True)

    if COMM.rank==0:
        print("unzipp", flush=True)
    h0,scale0, scale_var0 = zip(*merge0)
    h0 = COMM.gather(h0)
    scale0 = COMM.gather(scale0)
    scale_var0 = COMM.gather(scale_var0)
    if COMM.rank==0:
        print("unzipp", flush=True)
    h1, scale1, scale_var1 = zip(*merge1)
    h1 = COMM.gather(h1)
    scale1 = COMM.gather(scale1)
    scale_var1 = COMM.gather(scale_var1)

    if COMM.rank==0:
        print("stack", flush=True)
        h0 = np.vstack(h0)
        print("stack", flush=True)
        scale0 = np.hstack(scale0)
        print("stack", flush=True)
        scale_var0 = np.hstack(scale_var0)
        print("stack", flush=True)
        h1 = np.vstack(h1)
        print("stack", flush=True)
        scale1 = np.hstack(scale1)
        print("stack", flush=True)
        scale_var1 = np.hstack(scale_var1)

    if COMM.rank==0:
        print("broadcast", flush=True)
    h0 = COMM.bcast(h0)
    if COMM.rank==0:
        print("broadcast", flush=True)
    scale0 = COMM.bcast(scale0)
    if COMM.rank==0:
        print("broadcast", flush=True)
    scale_var0 = COMM.bcast(scale_var0)

    h1 = COMM.bcast(h1)
    scale1 = COMM.bcast(scale1)
    scale_var1 = COMM.bcast(scale_var1)

    if COMM.rank==0:
        print("zipping", flush=True)
    h0 = list(map(tuple, h0))
    merge0 = zip(h0, scale0, scale_var0)
    if COMM.rank==0:
        print("zipping", flush=True)
    h1 = list(map(tuple, h1))
    merge1 = zip(h1, scale1, scale_var1)
    #merge0 = COMM.bcast(COMM.reduce(merge0))
    #merge1 = COMM.bcast(COMM.reduce(merge1))
    if COMM.rank == 0:
        print("done reducing", flush=True)

    if COMM.rank == 0:
        print("groupby 0", flush=True)
    sort_key = lambda x: x[0]
    gb0 = groupby(sorted(merge0, key=sort_key), key=sort_key)
    merge_info0 = {k: [(I, I_var) for _, I, I_var in v] for k, v in gb0}

    if COMM.rank == 0:
        print("groupby 1", flush=True)
    gb1 = groupby(sorted(merge1, key=sort_key), key=sort_key)
    merge_info1 = {k: [(I, I_var) for _, I, I_var in v] for k, v in gb1}

    merge_results0 = []
    merge_results1 = []
    for i_chan, info in enumerate((merge_info0, merge_info1)):
        unique_hkl = set(info.keys())
        num_hkl = len(unique_hkl)
        for i_h, h in enumerate(unique_hkl):
            if i_h % COMM.size != COMM.rank:
                continue

            I, varI = map(np.array, zip(*info[h]))
            merge_output = merge_intensities(I, varI)
            if merge_output is None:
                continue
            mergeI, errorI, numI = merge_output
            if i_chan == 0:
                merge_results0.append((h, mergeI, errorI, numI))
            else:
                merge_results1.append((h, mergeI, errorI, numI))

            if COMM.rank == 0:
                print(
                    "Channel %d: merged successfully %1.4e, %1.4e, %d/%d" % (i_chan, mergeI, errorI, i_h + 1, num_hkl),
                    flush=True)

    merge_results0 = COMM.reduce(merge_results0)
    merge_results1 = COMM.reduce(merge_results1)

    if COMM.rank == 0:
        print("Saving results!")
        np.savez(outname, merge0=merge_results0, merge1=merge_results1)
        print("Donezo!")


def compute_completeness(merge_f, n_bins=15, log_f=None, channel=0,a=78.97, c=38.12):
    from cctbx import sgtbx, miller, crystal
    from dials.array_family import flex
    sym = crystal.symmetry((a,a,c,90,90,90), "P43212")
    h,_,_,_ = np.load(merge_f, allow_pickle=True)['merge%d' % channel].T
    h = np.vstack(h).astype(np.int32)
    hflex = flex.miller_index( list(map(tuple, h)))
    mset = miller.set(sym, hflex, True)
    mset.setup_binner(n_bins=n_bins)
    comp = mset.completeness(use_binning=True)
    print("COMPLETENESS for %s in channel %d:" % (merge_f,channel), file=log_f)
    comp.show(f=log_f)


if __name__ == "__main__":
    df = pandas.read_pickle(ARGS.input)
    overlaps = get_overlaps(df)
    n = len(df)
    m = n // 2

    # randomly divide into two halves
    inds = None
    if COMM.rank == 0:
        np.random.seed(8675349)
        inds = np.random.permutation(n)
    inds = COMM.bcast(inds)

    df1 = df.iloc[inds[:m]].copy().reset_index(drop=True)
    df2 = df.iloc[inds[m:]].copy().reset_index(drop=True)

    if COMM.rank == 0:
        if not os.path.exists(ARGS.outdir):
            os.makedirs(ARGS.outdir)
    COMM.barrier()

    even_f = os.path.join(ARGS.outdir, ARGS.prefix + "_even.npz")
    odd_f = os.path.join(ARGS.outdir, ARGS.prefix + "_odd.npz")
    all_f = os.path.join(ARGS.outdir, ARGS.prefix + "_all.npz")

    ## DO MERGES ##
    main(df1, even_f, overlaps)
    main(df2, odd_f, overlaps)
    if not ARGS.halvesOnly:
        main(df, all_f, overlaps)
    ## DONE MERGES ##

    if COMM.rank == 0:
        if not ARGS.halvesOnly:
            complete_logname = os.path.join(ARGS.outdir, "merge_all_complete.log" )
            with open(complete_logname, "w") as o:
                compute_completeness(all_f, ARGS.nbins, log_f=o, channel=0)
                compute_completeness(all_f, ARGS.nbins, log_f=o, channel=1)

        for i_chan in [0, 1]:
            d, cchalf, ccstar, dmin, dmax, overall_cc = cc_merge.cc_merge(even_f, odd_f, i_chan, ARGS.nbins)
            cc_table = cc_merge.get_table(dmin, dmax, cchalf, ccstar)
            cc_figname = os.path.join(ARGS.outdir, ARGS.prefix + "_cc_merge_chan%d.png" % i_chan)
            cc_merge.plot_cc(d, cchalf, ccstar, cc_figname)
            logname = os.path.join(ARGS.outdir, "merge_chan%d.log" % i_chan)
            with open(logname, "w") as o:
                compute_completeness(even_f, ARGS.nbins, log_f=o, channel=i_chan)
                compute_completeness(odd_f, ARGS.nbins, log_f=o, channel=i_chan)
                o.write("Working directory: %s\n" % os.getcwd())
                o.write("Command line input:\n %s\n" % " ".join(sys.argv))
                o.write(cc_table + "\n")
                o.write("Overall: CC1/2=%.2f, CC*=%.2f\n" % (overall_cc[0], overall_cc[1]))
