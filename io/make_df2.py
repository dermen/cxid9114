# coding: utf-8

from argparse import ArgumentParser
parser = ArgumentParser("Aggregates stage 1 results as a pandas dataframe, which is stored as a pandas pickle")
parser.add_argument("--nload", type=int, default=None, help="optional param to load a certain number of shots")
parser.add_argument("--filter", action="store_true", help="filter shots based on population statistics")
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

nsig = args.nsig
globstr = os.path.join( args.stage1dir, "results_job*/*trial2.pkl")
fnames = glob.glob(globstr)
if args.nload is not None:
    fnames = fnames[:args.nload]
df = pandas.concat([pandas.read_pickle(f) for f in fnames])
Norig = len(df)
cols = ["a", "c", "ncells", "spot_scales"]

df2 = df
# NOTE uncomment lines to filter
if args.filter:
    df2 = None
    for c in cols:
        m = df[c].median()
        mad = np.median( np.abs(df[c] - m) )
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
print("Ncells %10.7g" % (df2.ncells.median()))

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

