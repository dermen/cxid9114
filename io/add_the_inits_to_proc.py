# coding: utf-8
from argparse import ArgumentParser
parser = ArgumentParser("")
parser.add_argument("--glob", required=True, type=str)
parser.add_argument("--preopttag", default="preopt", type=str)
args = parser.parse_args()

import pandas
import h5py
import numpy as np
import glob

df_fnames = glob.glob(args.glob)
print("Concat %d dataframes..." % len(df_fnames))
df = pandas.concat([pandas.read_pickle(f) for f in df_fnames])

#df = pandas.read_pickle("procs0-29_inits.pkl")
unames = df.proc_fnames.unique()
for n in unames:
    dfn = df.query("proc_fnames=='%s'"%n)
    dfn_sort = dfn.sort_values(by='proc_shot_idx')
    Amats = np.array([ a for a in dfn_sort.Amats.values] )
    scales = dfn_sort.log_scales
    h5 = h5py.File(n, "r+")
    A = h5["Amatrices"]
    assert A.shape[0] == len(Amats)
    shot_ids = dfn_sort.proc_shot_idx.values
    assert all(shot_ids == np.arange( len(Amats)))
    
    k = "Amatrices_%s" % args.preopttag
    if k in h5:
        del  h5[k]
    h5.create_dataset(k, data=Amats)
    k = "crystal_scale_%s" % args.preopttag
    if k in h5:
        del h5[k]
    h5.create_dataset(k, data=scales)
    h5.close()
    print n
    
