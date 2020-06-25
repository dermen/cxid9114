
from argparse import ArgumentParser

parser = ArgumentParser("")
parser.add_argument("--datdir", type=str, required=True, help="path to directory containing diffBrag output files")
parser.add_argument("--binning", action="store_true")
parser.add_argument("--lastonly", action="store_true")
parser.add_argument("--comparefile", type=str, default=None)
parser.add_argument("--stride", type=int, default=1, help="stride over input files")
parser.add_argument("--truthpkl", type=str, required=True, help="truth miller array as a pickle")
parser.add_argument("--symbol", type=str, default="P43212")
parser.add_argument("--trial", type=int, default=None)
parser.add_argument("--unitcell", nargs=6, default=(79.1,79.1,38.4,90,90,90), type=float)
parser.add_argument("--dmin", type=float, default=2.125)
parser.add_argument("--savelast", type=str, default=None)
parser.add_argument("--mtzoutput", default=None, type=str, help="optional MTZ output filename, saves the structure factor amplitudes from last diffBragg iteration as an MTZ")
args = parser.parse_args()

import cctbx
from cctbx import miller
from cctbx.crystal import symmetry
import os
import numpy as np
import glob
from cxid9114 import utils
from cctbx.array_family import flex
from cctbx import sgtbx
from IPython import embed


symbol = args.symbol
ucell = args.unitcell
ftruth = utils.open_flex(args.truthpkl)


asu_map = np.load( os.path.join( args.datdir , "f_asu_map.npy") )[()]
fcell_pos, asu_indices = zip(*asu_map.items())

flex_indices = flex.miller_index(asu_indices)
if args.trial is None:
    data_files = glob.glob( os.path.join(args.datdir,  "_fcell_iter*.npz") )
else:
    data_files = glob.glob( os.path.join(args.datdir,  "_fcell_trial%d_iter*.npz" % args.trial) )

# order files by iteration number
#if args.trial is None:
data_files = sorted( data_files, key=lambda x: \
    int(os.path.basename(x).split("_iter")[1].split(".npz")[0]) )
#else:
#    data_files = sorted( data_files, key=lambda x: \
#        int(os.path.basename(x).split("_iter")[1].split(".npz")[0]) )

# get the starting and ending files
print (data_files)
n_files = len(data_files)

# always load the first and last iteration.. 
indices = [0]+ range( 1, n_files-2, args.stride) + [n_files-1]
if args.lastonly:
    indices = [indices[-1]]

sgi = sgtbx.space_group_info(symbol)
sym = symmetry(unit_cell=ucell, space_group_info=sgi)

mset = miller.set(sym , flex_indices, anomalous_flag=True)
for i_iter in indices:
    fcell_data = flex.double( np.load(data_files[i_iter])['fvals']  )
    fobs = miller.array( mset, data=fcell_data).set_observation_type_xray_amplitude()
    
    if args.savelast is not None and i_iter==indices[-1]:
        utils.save_flex(fobs, args.savelast)
        
    
    fobs_sel = fobs.select(fobs.resolution_filter_selection(d_max=30, d_min=args.dmin))
    if args.binning:
        r,c = utils.compute_r_factor_binned( ftruth, fobs_sel, verbose=True, d_max=30, d_min=args.dmin, n_bin=10)
        r,c = utils.compute_r_factor( ftruth, fobs_sel, verbose=False, is_flex=True, sort_flex=True)
        print ("Iter %d: Overall: Rtruth=%.4f, CCdeltaAnom=%.4f"% (i_iter, r,c))
    else:
        r,c = utils.compute_r_factor( ftruth, fobs_sel, verbose=False, is_flex=True, sort_flex=True)
        print ("iter %d: Rtruth=%.4f, CCdeltaAnom=%.4f"% (i_iter, r,c))
    print fobs_sel.completeness()


if args.mtzoutput is not None:
    fobs.as_mtz_dataset("Fobs").mtz_object().write(args.mtzoutput)

if args.comparefile is not None:
    print("<><><><><><><><><><><><><><>")
    print("Comparison to integration file %s:" % args.comparefile)
    print("<><><><><><><><><><><><><><>")
    from iotbx.reflection_file_reader import any_reflection_file
    F = any_reflection_file(args.comparefile).as_miller_arrays()[0]
    F = F.as_amplitude_array()
    fmap = {i:d for i,d in zip(F.indices(), F.data())}
    Fdata = []
    for i in fobs_sel.indices():
        if i in fmap:
            Fdata.append(fmap[i])
        else:
            Fdata.append(0)
    #Fdata = [fmap[i] for i in fobs.indices() ]
    mset = miller.set(sym, indices=fobs_sel.indices(), anomalous_flag=True)
    Fobs = miller.array( mset, data=flex.double(Fdata)).set_observation_type_xray_amplitude()
    r,c = utils.compute_r_factor_binned( ftruth, Fobs, verbose=True, d_max=30, d_min=args.dmin, n_bin=10)
    r,c = utils.compute_r_factor( ftruth, Fobs, verbose=False, is_flex=True, sort_flex=True) 
    print ("Overall : Rtruth=%.4f, CCdeltaAnom=%.4f"% (r,c))


