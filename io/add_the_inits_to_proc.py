# coding: utf-8
from argparse import ArgumentParser
parser = ArgumentParser("")
parser.add_argument("--glob", required=True, type=str)
parser.add_argument("--preopttag", default="preopt", type=str)
parser.add_argument("--bgextracted", action="store_true")
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
    #dfn_sort = dfn.sort_values(by='proc_shot_idx')
    
    #Amats = np.array([ a for a in dfn_sort.Amats.values] )
    #scales = dfn_sort.spot_scales
    #ncells = dfn_sort.ncells
    
    h5 = h5py.File(n, "r+")
    nshots = h5["Amatrices"].shape[0]
    #assert A.shape[0] == len(Amats)
    #shot_ids = dfn_sort.proc_shot_idx.values
    #assert all(shot_ids == np.arange( len(Amats)))
    # add in the optimized background planes
    Amats =[]
    scales = []
    ncells = []
    bg_coefs = []
    for i_shot in range(nshots):
        df_shot = dfn.query("proc_shot_idx==%d"%i_shot)
        
        if not args.bgextracted:
            tilt_abc = h5["tilt_abc"]["shot%d"%i_shot][()]
            bbox_idx = df_shot.bbox_idx.values[0]
            abc = df_shot.bgplanes.values[0]
            for ii, bb_i in enumerate(bbox_idx):
                tilt_abc[bb_i] = abc[ii]
            k = "tilt_abc_%s/shot%d" % (args.preopttag, i_shot)
            if k in h5:
                del h5[k]
            h5.create_dataset(k, data=tilt_abc)
        else:
            bg_coefs.append(df_shot.bg_coef.values[0])

        Amats.append( df_shot.Amats.values[0])
        ncells.append( df_shot.ncells.values[0] )
        scales.append( df_shot.spot_scales.values[0])
    # end add in the optimized background planes    
   
    if args.bgextracted:
        k = "background_coefficients_%s" % args.preopttag
        if k in h5:
            del  h5[k]
        h5.create_dataset(k, data=bg_coefs)
    
    k = "Amatrices_%s" % args.preopttag
    if k in h5:
        del  h5[k]
    h5.create_dataset(k, data=Amats)
    
    k = "spot_scale_%s" % args.preopttag
    if k in h5:
        del h5[k]
    h5.create_dataset(k, data=scales)
    
    k = "ncells_%s" % args.preopttag
    if k in h5:
        del h5[k]
    h5.create_dataset(k, data=ncells)
    
    h5.close()
    print n
    
