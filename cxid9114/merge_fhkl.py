
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("input", type=str, help="input concat pandas pickles from hopper run with refine.Fhkl=True")
parser.add_argument("outdir", type=str, help="output folder")
parser.add_argument("--prefix", type=str, help="output file prefix", default="mergy-poo")
parser.add_argument("--thresh", default=5, type=float, help="outlier intensity threshold (amongst ASU equivalents")
parser.add_argument("--maxProc", default=None, type=int),
parser.add_argument("--nbins", default=15, type=int, help="number of resolution bins for merge stats")
parser.add_argument("--minSNR", default=0, type=float, help="min signal to noise for a spot")
parser.add_argument("--minVar", default=1e-7, type=float, help="min variance for spot to merge")
parser.add_argument("--maxVar", default=1e12, type=float, help="max variance for spot to merge")
parser.add_argument("--minMeas", default=3, type=int, help="minimum multiplicity")
parser.add_argument("--weighted", action="store_true")
parser.add_argument("--median", action="store_true")
parser.add_argument("--maxSigZ", type=float, default=2.1)
parser.add_argument("--minSigZ", type=float, default=0.8)
parser.add_argument("--minNiter", type=int, default=500)
parser.add_argument("--reference", type=str, default=None, help="path to .npz from previous merge")
parser.add_argument("--halvesOnly", action="store_true")
ARGS = parser.parse_args()

import pandas
from cctbx import miller, crystal
from scipy.stats import linregress
from dials.array_family import flex
import os
from simtbx.diffBragg import utils
import numpy as np
import sys
from simtbx.diffBragg.utils import ENERGY_CONV
from scipy.optimize import minimize
from itertools import groupby
from libtbx.mpi4py import MPI
from cxid9114 import cc_merge

COMM = MPI.COMM_WORLD


def print0(*args, **kwargs):
    if COMM.rank==0:
        #if "flush" in list(kwargs.keys()):
        #    kwargs['flush'] = True
        print(*args, **kwargs)


def scale_to_reference(merge_data, reference_data, i_chan=0, how='lsq'):

    data_h, data_I, data_varI = zip(*merge_data)
    ref_I = [reference_data[i_chan][h] for h in data_h if h in reference_data[i_chan]]
    data_in_ref_I = [data_I[i_h] for i_h, h in enumerate(data_h) if h in reference_data[i_chan]]

    ref_I = np.array(ref_I)
    data_in_ref_I = np.array(data_in_ref_I)

    sel = ~utils.is_outlier(data_in_ref_I, )

    if how=='nelder-mead':
        def func_min(p, ref_I, data_I):
            scale = p[0]
            resid = ref_I - scale * data_I
            return np.sum(resid ** 2)

        out = minimize(func_min, x0=np.array([1]), args=(ref_I[sel], data_in_ref_I[sel]), method='Nelder-Mead')
        if not out.success:
            # fall back to least sq
            scale = (data_in_ref_I * ref_I)[sel].sum() / (data_in_ref_I ** 2)[sel].sum()
        else:
            scale = out.x[0]

        if scale<0:
            print("Bad scale factor! Zeroing out shot")
            scale = 0
    elif how=='lsq':
        scale = (data_in_ref_I*ref_I)[sel].sum() / (data_in_ref_I**2)[sel].sum()
    else:
        scale = 1

    data_h = [h for i_h, h in enumerate(data_h) if sel[i_h]]
    data_I = [scale*d for i_d, d in enumerate(data_I) if sel[i_d]]
    data_varI = [scale**2*d for i_d, d in enumerate(data_varI) if sel[i_d]]
    merge_data = list(zip(data_h, data_I, data_varI))
    return merge_data


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

    if len(I) == 1:
        inliers = np.ones(1, bool)
    else:
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
                print0("basename %s has %d xtals and %d (%.2f%%) bad hkls= " % (g, nxtal, nbad, frac_bad), overlap_hkls)
            overlaps.append( [g, overlap_hkls])

    overlaps = COMM.reduce(overlaps)
    if COMM.rank==0:
        overlaps = {g: hkls for g,hkls in overlaps}
    overlaps = COMM.bcast(overlaps)
    return overlaps


def main(df, outname, overlaps, reference=None):
    mtz_name = "/global/cfs/cdirs/lcls/dermen/d9114_data/merging/merge2/iobs_all.mtz"
    mtz_column = "Iobs,SIGIobs"
    Famp = utils.open_mtz(mtz_name, mtz_column)
    Famp = Famp.expand_to_p1()
    Famp = Famp.generate_bijvoet_mates()

    Flookup = {h: val for h, val in zip(Famp.indices(), Famp.data())}

    reference_data = None
    if reference is not None:
        reference_data = {}
        for i_chan in [0,1]:
            ref_h, ref_I, _,_ = np.load(reference, allow_pickle=True)["merge%d" % i_chan].T
            reference_data[i_chan] = {val_h: val_I for val_h, val_I in zip(ref_h, ref_I)}

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
            if reference_data is not None:
                merge_data = scale_to_reference(merge_data, reference_data, i_chan)
            if i_chan == 0:
                merge0 += merge_data
            else:
                merge1 += merge_data
        # if i_f==200#:
        #    break
        print0("Loaded and corrected intensities %d/%d" % (i_f + 1, len(df)), flush=True)

    print0("reducing", flush=True)

    print0("unzipp", flush=True)
    h0,scale0, scale_var0 = zip(*merge0)
    h0 = COMM.gather(h0)
    scale0 = COMM.gather(scale0)
    scale_var0 = COMM.gather(scale_var0)
    print0("unzipp", flush=True)
    h1, scale1, scale_var1 = zip(*merge1)
    h1 = COMM.gather(h1)
    scale1 = COMM.gather(scale1)
    scale_var1 = COMM.gather(scale_var1)

    if COMM.rank==0:
        print0("stack", flush=True)
        h0 = np.vstack(h0)
        print0("stack", flush=True)
        scale0 = np.hstack(scale0)
        print0("stack", flush=True)
        scale_var0 = np.hstack(scale_var0)
        print0("stack", flush=True)
        h1 = np.vstack(h1)
        print0("stack", flush=True)
        scale1 = np.hstack(scale1)
        print0("stack", flush=True)
        scale_var1 = np.hstack(scale_var1)

    print0("broadcast", flush=True)
    h0 = COMM.bcast(h0)
    print0("broadcast", flush=True)
    scale0 = COMM.bcast(scale0)
    print0("broadcast", flush=True)
    scale_var0 = COMM.bcast(scale_var0)

    h1 = COMM.bcast(h1)
    scale1 = COMM.bcast(scale1)
    scale_var1 = COMM.bcast(scale_var1)

    print0("zipping", flush=True)
    h0 = list(map(tuple, h0))
    merge0 = zip(h0, scale0, scale_var0)
    print0("zipping", flush=True)
    h1 = list(map(tuple, h1))
    merge1 = zip(h1, scale1, scale_var1)
    #merge0 = COMM.bcast(COMM.reduce(merge0))
    #merge1 = COMM.bcast(COMM.reduce(merge1))
    print0("done reducing", flush=True)

    print0("groupby 0", flush=True)
    sort_key = lambda x: x[0]
    gb0 = groupby(sorted(merge0, key=sort_key), key=sort_key)
    merge_info0 = {k: [(I, I_var) for _, I, I_var in v] for k, v in gb0}

    print0("groupby 1", flush=True)
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

            print0(
                "Channel %d: merged successfully %1.4e, %1.4e, %d/%d" % (i_chan, mergeI, errorI, i_h + 1, num_hkl),
                flush=True)

    merge_results0 = COMM.reduce(merge_results0)
    merge_results1 = COMM.reduce(merge_results1)

    print0("Saving results!")
    if COMM.rank == 0:
        np.savez(outname, merge0=merge_results0, merge1=merge_results1)
    print0("Donezo!")


def compute_completeness(merge_f, n_bins=15, log_f=None, channel=0,a=78.97, c=38.12):
    sym = crystal.symmetry((a,a,c,90,90,90), "P43212")
    h,_,_,_ = np.load(merge_f, allow_pickle=True)['merge%d' % channel].T
    h = np.vstack(h).astype(np.int32)
    hflex = flex.miller_index( list(map(tuple, h)))
    mset = miller.set(sym, hflex, True)
    mset.setup_binner(n_bins=n_bins)
    comp = mset.completeness(use_binning=True)
    print0("COMPLETENESS for %s in channel %d:" % (merge_f,channel), file=log_f)
    comp.show(f=log_f)


def convert_to_mtz(merge_f, a=78.97, c=38.2):
    sym = crystal.symmetry((a,a,c,90,90,90), "P43212")
    mtz_name = merge_f.replace(".npz", "_channel%d.mtz")

    for i_chan in [0,1]:
        if i_chan==0:
            wave = ENERGY_CONV / 8944.
        else:
            wave = ENERGY_CONV / 9034

        h,I,errorI,mult = np.load(merge_f, allow_pickle=True )["merge%d" %i_chan].T
        I = np.ascontiguousarray(I).astype(np.float64)
        errorI = np.ascontiguousarray(errorI).astype(np.float64)
        h = np.vstack(h).astype(np.int32)
        hflex = flex.miller_index(list(map(tuple, h)))
        mset = miller.set(sym, hflex, True)
        ma = miller.array(mset, data=flex.double(I), sigmas=flex.double(errorI))
        ma = ma.set_observation_type_xray_intensity()
        ma.as_mtz_dataset(column_root_label="I", wavelength=wave).mtz_object().write(mtz_name % i_chan)


def merge_trial(outdir, prefix, reference=None, halvesOnly=False, nbins=15):
    even_f = os.path.join(outdir, prefix + "_even.npz")
    odd_f = os.path.join(outdir, prefix + "_odd.npz")
    all_f = os.path.join(outdir, prefix + "_all.npz")

    ## DO MERGES ##
    main(df1, even_f, overlaps, reference)
    main(df2, odd_f, overlaps, reference)

    if COMM.rank==0:
        for i_chan in [0, 1]:
            d, cchalf, ccstar, dmin, dmax, overall_cc = cc_merge.cc_merge(even_f, odd_f, i_chan, nbins)
            cc_table = cc_merge.get_table(dmin, dmax, cchalf, ccstar)
            cc_figname = os.path.join(outdir, prefix + "_cc_merge_chan%d.png" % i_chan)
            cc_merge.plot_cc(d, cchalf, ccstar, cc_figname)
            logname = os.path.join(outdir, "merge_chan%d.log" % i_chan)
            with open(logname, "w") as o:
                compute_completeness(even_f, nbins, log_f=o, channel=i_chan)
                compute_completeness(odd_f, nbins, log_f=o, channel=i_chan)
                o.write("Working directory: %s\n" % os.getcwd())
                o.write("Command line input:\n %s\n" % " ".join(sys.argv))
                o.write(cc_table + "\n")
                o.write("Overall: CC1/2=%.2f, CC*=%.2f\n" % (overall_cc[0], overall_cc[1]))

    if not halvesOnly:
        main(df, all_f, overlaps, reference)
        if COMM.rank == 0:
            complete_logname = os.path.join(outdir, "merge_all_complete.log" )
            with open(complete_logname, "w") as o:
                compute_completeness(all_f, nbins, log_f=o, channel=0)
                compute_completeness(all_f, nbins, log_f=o, channel=1)
            convert_to_mtz(all_f)


if __name__ == "__main__":
    if ARGS.maxProc is not None:
        df = pandas.read_pickle(ARGS.input).iloc[:ARGS.maxProc]
    else:
        df = pandas.read_pickle(ARGS.input)

    print0("Before filtering sigz,niter,null: number of records=%d" % len(df))
    df = df.loc[df.sigz.notnull()].query("%f < sigz < %f" %(ARGS.minSigZ, ARGS.maxSigZ)).query("niter>%f" % ARGS.minNiter).reset_index(drop=True)
    print0("After filtering sigz,niter,null: number of records=%d" % len(df))
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
    merge_trial(ARGS.outdir, ARGS.prefix, reference=ARGS.reference, halvesOnly=ARGS.halvesOnly, nbins=ARGS.nbins)
