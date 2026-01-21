# coding: utf-8
# TODO:make more general for different unit cells
from argparse import ArgumentParser
parser = ArgumentParser("Load and refine bigz")
parser.add_argument("--glob", type=str, required=True, help="glob for selecting files (output files of process_filtered_mpi)")
parser.add_argument("--truth", type=str, default=None, help="path to pickled miller array of ground truth")
parser.add_argument("--merge", type=str, required=True, help="path to input merge (pkl of miller array or mtz)")
parser.add_argument("--out", type=str, required=True, help='output filename')
parser.add_argument("--mtzoutput", action='store_true', help='write an mtz file as output')
parser.add_argument("--mtzinput", action='store_true', help='input merge file is an mtz file')
parser.add_argument("--dhighres", type=float, default=2, help="high resolution limit")
parser.add_argument("--symbol", type=str, default="P43212")
parser.add_argument("--unitcell", nargs=6, type=float, default=None,
                    help="space separated unit cell e.g. --unitcell 79 79 38 90 90 90")
args = parser.parse_args()

from simtbx.diffBragg.utils import map_hkl_list
import h5py
import glob
import numpy as np
from cxid9114 import utils
from scipy.spatial import cKDTree
from iotbx.reflection_file_reader import any_reflection_file
from cctbx.array_family import flex
from cctbx import miller
from cctbx.crystal import symmetry
from cxid9114 import parameters


a, b, c, al, be, ga = args.unitcell

# load all miller indices corresponding to diffBragg ROIs
fnames = glob.glob(args.glob)
all_Hi = []
#all_res = []
for f in fnames:
    h5 = h5py.File(f, 'r')
    shots = h5["Hi"].keys()
    Hi = np.vstack([h5["Hi"][s] for s in shots])
    #res = np.hstack([h5["resolution"][s] for s in shots])
    all_Hi.extend(list(map(tuple, Hi)))
    #all_res.append(res)
    print (f, len(shots))

# unique asu Hi:
u_all_Hi = list(set(map_hkl_list(all_Hi, symbol=args.symbol)))

# Load the cctbx.xfel.merge output for estimation of initial params    
if args.mtzinput:
#   TODO: make this so that one can selec on label (e.g. fobs)
    F = any_reflection_file(args.merge).as_miller_arrays()[0].as_amplitude_array()
else:
    F = utils.open_flex(args.merge)#.as_amplitude_array()
Fmap = {h: amp for h, amp in zip(F.indices(), F.data())}
merge_Hi = list(set(map_hkl_list(Fmap.keys(), symbol=args.symbol)))

symm = symmetry(unit_cell=(a, b, c, al, be, ga), space_group_symbol=args.symbol)

# diffBragg resolutions
mset = miller.set(symm, flex.miller_index(u_all_Hi), anomalous_flag=True)
reso = mset.d_spacings().data()
reso = np.array(reso)[:,None]

# merged resolutions
msetm = miller.set(symm, flex.miller_index(merge_Hi), anomalous_flag=True)
resom = msetm.d_spacings().data()
resom = np.array(resom)[:, None]

tree = cKDTree(resom)
o, knearest = tree.query(reso, k=5)

u_all_amp = []
for i_h, h in enumerate(u_all_Hi):
    if h in Fmap:
        in_array = True
        amp = Fmap[h]
    else:
        in_array = False
        if np.any(np.isinf(o[i_h])):
            amp = 0
        else:
            amps = [Fmap[merge_Hi[ii]] for ii in knearest[i_h]]
            amp = np.median(amps)
    u_all_amp.append(amp)

bad_idx = u_all_Hi.index((0, 0, 0))
u_all_Hi.pop(bad_idx)
u_all_amp.pop(bad_idx)

indices = flex.miller_index(u_all_Hi)
data = flex.double(u_all_amp)

# save as an MTZ
mset = miller.set(symm, indices=indices, anomalous_flag=True)
marray = miller.array(mset, data).set_observation_type_xray_amplitude()
if args.mtzoutput:
    mtz = marray.as_mtz_dataset(column_root_label='fobs', wavelength=parameters.WAVELEN_HIGH)
    ob = mtz.mtz_object()
    ob.write(args.out)
# save as a PKL
else:
    utils.save_flex( marray, args.out)

if args.truth is not None:
    Ftruth = utils.open_flex(args.truth)
    marray_2 = marray.select(marray.resolution_filter_selection(d_max=30, d_min=args.dhighres))
    r,c = utils.compute_r_factor(Ftruth, marray_2, is_flex=True, sort_flex=True)
    print("Truth R-factor=%4f and CC-Delta-anom=%.4f out to %.4f Angstrom" % (r,c, args.dhighres))
