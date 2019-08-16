# coding: utf-8
import glob
import os
import h5py
from copy import deepcopy

from dxtbx.model.experiment_list import ExperimentListFactory

from cxid9114 import utils

from dials.algorithms.indexing.compare_orientation_matrices \
    import rotation_matrix_differences, difference_rotation_matrix_axis_angle

# reflecton file names
fnames = glob.glob("job*/refls*/refl*pkl")

refls_good,Els_good, dumps_good,cryst_good = [],[],[],[]
all_rots = []
for i_f, f in enumerate(fnames): # [:100]):
    El_f = f.replace("refl", "El")
    dump_f = f.replace("refl", "dump")
    if os.path.exists(El_f) and os.path.exists(dump_f):
        if i_f %40 ==0:
            print("Opened %d / %d indexing results" % (i_f, len(fnames)))
        refls_good.append( f)
        Els_good.append(El_f)
        dumps_good.append(dump_f)
        
        idx_cryst_mdl = utils.open_flex(dump_f)["crystalAB"]
        cryst_good.append(idx_cryst_mdl)
        Ex = ExperimentListFactory.from_json_file(El_f, check_format=False)
        img_path = Ex[0].imageset.get_path(0)
        h5 = h5py.File(img_path, "r")
        Atruth = h5["crystalA"][()]
        
        truth_cryst = deepcopy(idx_cryst_mdl)
        truth_cryst.set_A(Atruth)
        
        out = difference_rotation_matrix_axis_angle(idx_cryst_mdl, truth_cryst)
        all_rots.append(out[2])


from IPython import embed
embed()

"""

Els_good
len( refls_good)
len( Els_good)
import dxtbx
ff
E = ff[0]
E.imageset
E.imageset.get_path()
E.imageset.get_path
E.imageset.get_path()
E.imageset.get_path(0)
E.imageset.get_path(0)
El_path = E.imageset.get_path(0)
h5.keys()
h5["crystalA"]
Ctruth = h5["crystalA"]
Ctruth
from copy import deepcopy
deepcopy(Ctruth)
get_ipython().magic(u'timeit deepcopy(Ctruth)')
refls_good,Els_good, dumps_good = [],[],[]
for f in fnames:
    El_f = f.replace("refl", "El")
    dump_f = f.replace("refl", "dump")
    if os.path.exists(El_f) and os.path.exists(dump_f):
        refls_good.append( f)
        Els_good.append(El_f)
        dumps_good.append(dump_f)
        
dumps_good
utils.open_flex(dumps_good[0])
get_ipython().magic(u'timeit utils.open_flex(dumps_good[0])["crystalAB"]')
get_ipython().magic(u'timeit utils.open_flex(dumps_good[0])["crystalAB"]')
get_ipython().magic(u'timeit utils.open_flex(dumps_good[0])["crystalAB"]')
#utils.open_flex(dumps_good[0])["crystalAB"]

cryst_good
deepcopy(cryst_good[0])
get_ipython().magic(u'timeitdeepcopy (cryst_good[0])')
get_ipython().magic(u'timeitd eepcopy(cryst_good[0])')
get_ipython().magic(u'timeit deepcopy(cryst_good[0])')
get_ipython().magic(u'save orient_mats.py 1-59')
"""
