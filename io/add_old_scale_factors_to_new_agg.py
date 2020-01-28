# coding: utf-8
from argparse import ArgumentParser
parser = ArgumentParser("")
parser.add_argument("--pklglob", required=True, type=str)
parser.add_argument("--aggglob", required=True, type=str)
parser.add_argument("--preopttag", default="preopt", type=str)
args = parser.parse_args()

import pandas
import h5py
import numpy as np
import glob
import os


def get_basename(img_path, img_file_suffix=".h5.npz"):
    basename = os.path.basename(img_path)
    # TODO: make more general with a suffix arg
    basename = basename.split(img_file_suffix)[0]
    return basename


df_fnames = glob.glob(args.pklglob)
print("Concat %d dataframes..." % len(df_fnames))
df = pandas.concat([pandas.read_pickle(f) for f in df_fnames])

# these are the files in the new database 
new_proc_fnames = glob.glob(args.aggglob)

# these are the files where we refined the scale factor
old_proc_path = {f:h5py.File(f,'r')['h5_path'] for f in df.proc_fnames.unique()}

img_basenames = []
scale_data = {}
Amat_data = {}
stuff = ['proc_fnames', 'proc_shot_idx', "log_scales", "Amats"]
for old_fname, old_idx, log_scale, A in df[stuff].values:
    img_path = old_proc_path[old_fname][old_idx]
    img_basename = get_basename(img_path)
    scale_data[img_basename] = log_scale
    Amat_data[img_basename] = A
    
for proc_fname in new_proc_fnames:
    h5 = h5py.File(proc_fname, 'r+')
    basenames = [get_basename(img_path) for img_path in h5["h5_path"][()]]
    
    log_scales, Amats = [], []
    for basename in basenames:
        log_scales.append(scale_data[basename])
        Amats.append( Amat_data[basename])
    
    k = "Amatrices_%s" % args.preopttag
    if k in h5:
        del  h5[k]
    h5.create_dataset(k, data=Amats)
    k = "crystal_scale_%s" % args.preopttag
    if k in h5:
        del h5[k]
    h5.create_dataset(k, data=log_scales)
    h5.close()
    print proc_fname 

