# coding: utf-8
from argparse import ArgumentParser

parser = ArgumentParser("")
parser.add_argument("--glob", type=str, required=True, help="path to pandas pickle files as a glob")
parser.add_argument("--expf", type=str, required=True, help="path to filtered experiment list (or combined)")
parser.add_argument("--reff", type=str, required=True, help="path to filtered reflection table (or combined)")
parser.add_argument("--out", type=str, default="refined", help="output file name (will write out.refl and out.expt)")
parser.add_argument("--thresh", type=float, default=3.5, help="outlier threshold for image correlation")
args = parser.parse_args()


import pandas
import glob
import h5py
import os
from dxtbx.model.crystal import Crystal
from dxtbx.model.experiment_list import ExperimentList
from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory
from copy import deepcopy
from cxid9114 import utils

print("Loading pandas pickle")

all_df = pandas.concat( [pandas.read_pickle(f) for f in glob.glob(args.glob)])
#df = pandas.read_pickle("refine_kaladin_2_gpu_refined_3.pkl")

print ("I found data on  %d experiments" % len(all_df))

is_bad_expt = utils.is_outlier(all_df.image_corr, args.thresh)
df = all_df.loc[~is_bad_expt]

df_bad = all_df.loc[is_bad_expt]
print ("I removed %d experiments that were apparently diverging" % len(df_bad)) 
print ("Average model correlation of kept experiments= %.3f" % df.image_corr.mean())
print ("Average model correlation of removed experiments= %.3f" % df_bad.image_corr.mean())


print("Loading Amatrices optimized from the agg file")
u_proc_fnames = {f:h5py.File(f, 'r') for f in df.proc_fnames.unique()}
#Amat = { fname: h5["Amatrices_preopt2"] for fname, h5 in u_proc_fnames.items()}
img_paths = { fname: h5["h5_path"] for fname, h5 in u_proc_fnames.items()}

print ("Adding basenames to the dataframe ") 
basenames = []
#newA = []
for fname, idx in df[["proc_fnames", "proc_shot_idx"]].values:
    #A = Amat[fname][idx]
    img_path = img_paths[fname][idx]
    basename = os.path.basename(img_path).split(".h5.npz")[0]
    basenames.append(basename)
    #newA.append( tuple(A))

df["basename"] = basenames
#df["newA"] = newA

print ("Opening experiment list file")
from dials.array_family import flex
refl = flex.reflection_table.from_file(args.reff)
El = ExperimentListFactory.from_json_file(args.expf, check_format=False)
print("Iterating the experiment lists and updating the experiment crystals with optimal A ")
n_exper = len(El)
new_refl = flex.reflection_table() 
new_expl = ExperimentList()

for i_exp in range(n_exper):
   
    exper_img_path = El[i_exp].imageset.get_path(0)
    exper_basename = os.path.basename( exper_img_path)
    exper_basename = exper_basename.split(".h5.npz")[0]
    exper_crystal = El[i_exp].crystal
    new_crystal = deepcopy(exper_crystal)
    df_sel = df.query("basename=='%s'"% exper_basename)
    if not len(df_sel)==1:
        continue
    new_A = df_sel["Amats"].values[0]
    new_crystal.set_A(new_A)
    a,_,c,_,_,_ = new_crystal.get_unit_cell().parameters()
    if i_exp % 50==0:
        print ("Processed %d / %d experiments" % (i_exp+1, n_exper))
    El[i_exp].crystal = new_crystal

    # save the experiemnts
    new_expl.append( El[i_exp]) 
    refl_i = refl.select(refl['id']==i_exp) # NOTE: this might change at some point... 
    new_refl.extend(refl_i)

new_refl.as_file(args.out + ".refl")
new_expl.as_file(args.out + ".expt")

    
