# coding: utf-8

from argparse import ArgumentParser
parser = ArgumentParser("Aggregates stage 1 results as a pandas dataframe, which is stored as a pandas pickle")
parser.add_argument("--nload", type=int, default=None, help="optional param to load a certain number of shots")
parser.add_argument("--filter", action="store_true", help="filter shots based on population statistics")
parser.add_argument("--addMasterInfo", action="store_true", help="add master file information to files")
parser.add_argument("--plot", action="store_true", help="whether to display histograms of various parameters")
parser.add_argument("--stage1dir", type=str, help="path to diffBragg stage1 output folder", required=True)
parser.add_argument("--nsig", default=7, type=float, help="parameter for filtering shots (decrease to filter mode)")
parser.add_argument("--pklname", type=str, default=None, 
    help="optional oupput name for pandas pickle file (default is /stage1dir/df2.pkl")
args = parser.parse_args()

import pandas
import glob
import os
import pylab as plt
import numpy as np
import sys
import h5py

nsig = args.nsig
globstr = os.path.join( args.stage1dir, "results_job*/*trial2.pkl")
fnames = glob.glob(globstr)
if args.nload is not None:
    fnames = fnames[:args.nload]
df = pandas.concat([pandas.read_pickle(f) for f in fnames])

if args.addMasterInfo:
    print("adding master file info")
    u_fnames = df.proc_fnames.unique()
    u_h5s = {f:h5py.File(f,'r')["h5_path"][()] for f in u_fnames}
    u_master_indices = {f:h5py.File(f,'r')["master_file_indices"][()] for f in u_fnames}
    img_fnames = []
    master_indices = []
    for f, idx in df[['proc_fnames', 'proc_shot_idx']].values:
        img_fnames.append(u_h5s[f][idx])
        master_indices.append(u_master_indices[f][idx])
    df["imgpaths"] = img_fnames
    df["master_indices"] = master_indices


Norig = len(df)

try:
    n_ncells_param = len(df["ncells"].values[0])
    ncells_vals = list(zip(*df.ncells.values))
except TypeError:
    n_ncells_param = 1
    ncells_vals = list(df.ncells.values)
ncells_cols = []
for i_ncells in range(n_ncells_param):
    col_name = "ncells%d" %i_ncells
    ncells_cols.append(col_name)
    df[col_name] = ncells_vals[i_ncells]

cols = ["a", "c", "spot_scales"] + ncells_cols

df2 = df
if args.filter:
    df2 = None
    for c in cols:
        m = df[c].median()
        mad = np.median( np.abs(df[c] - m))
        query = "%f < %s < %f" % (m-nsig*mad, c, m+nsig*mad )
        print(query)
        if df2 is None:
            df2 = df.query(query)
        else:
            df2 = df2.query(query)

print("Maximum misorient = %f" % df2.final_misorient.max())
print("Median misorient = %f" % df2.final_misorient.median())
print("Ucell %f %10.7g" % (df2.a.median(), df2.c.median()))
print("Spotscale %10.7g" % (df2.spot_scales.median()))
for col in ncells_cols:
    print("%s %10.7g" % (col, df2[col].median()))

pklname = args.pklname
if args.pklname is None:
    pklname = os.path.join( args.stage1dir, "df2.pkl")

df2.to_pickle(pklname)
print("Filtered pickle %s contains data on %d/%d images" % (pklname, len(df2), Norig))
if args.plot:
    for c in cols + ["final_misorient"]:
        plt.figure()
        vals = list(df[c].values) + list(df2[c].values)
        bins = np.logspace(np.log10(min(vals)), np.log10(max(vals)), 100)
        plt.hist(df[c], bins=bins, log=1, histtype='step', lw=2)
        plt.hist(df2[c], bins=bins, log=1, histtype='step', lw=2) 
        plt.legend(("%d shots" % len(df),"%d shots" % len(df2)))
    plt.show() 

