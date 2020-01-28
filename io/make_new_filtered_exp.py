# coding: utf-8
import pandas
import h5py
import os
from dxtbx.model.crystal import Crystal
from dxtbx.model.experiment_list import ExperimentList
from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory
from copy import deepcopy

print("Loading pandas pickle")
df = pandas.read_pickle("refine_kaladin_2_gpu_refined_3.pkl")

print("Loading Amatrices optimized from the agg file")
u_proc_fnames = {f:h5py.File(f, 'r') for f in df.proc_fnames.unique()}
Amat = { fname: h5["Amatrices_preopt2"] for fname, h5 in u_proc_fnames.items()}
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
El = ExperimentListFactory.from_json_file("/global/cscratch1/sd/dermen/kaladin_2/filtered.expt", check_format=False)
print("Iterating the experiment lists and updating the experiment crystals with optimal A ")
n_exper = len(El)
    
for i_exp in range(n_exper):
    exper_img_path = El[i_exp].imageset.get_path(0)
    exper_basename = os.path.basename( exper_img_path)
    exper_basename = exper_basename.split(".h5.npz")[0]
    exper_crystal = El[i_exp].crystal
    new_crystal = deepcopy(exper_crystal)
    df_sel = df.query("basename=='%s'"% exper_basename)
    assert len(df_sel)==1
    new_A = df_sel["Amats"].values[0]
    new_crystal.set_A(new_A)
    a,_,c,_,_,_ = new_crystal.get_unit_cell().parameters()
    print (a,c, len(new_A))
    El[i_exp].crystal = new_crystal

from IPython import embed
embed()

    
