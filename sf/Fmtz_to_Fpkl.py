# coding: utf-8
from argparse import ArgumentParser

parser = ArgumentParser("convert mtz to pkl and compare with ground truth")
parser.add_argument("--input", type=str,required=True, help="name of input mtz file from cctbx.xfel.merge")
parser.add_argument("--output", type=str,required=True, help="name of output pkl file")
parser.add_argument("--reference", type=str, required=True, help="|F| reference pickle")
parser.add_argument("--referenceForCCano", type=str, required=True, help="|F| reference pickle, for computing CCano (only heavy atoms should have anomalous scatering corrections in this reference)")
args = parser.parse_args()

from iotbx.reflection_file_reader import any_reflection_file
from cxid9114 import utils
F = any_reflection_file(args.input).as_miller_arrays()[0]
F = F.as_amplitude_array()
utils.save_flex( F, args.output)
print("wrote amplitudes from mtz as a cctbx miller array in file %s " % args.output)

# compare with ground truth
ftruth = utils.open_flex(args.reference)  # for computing R with ground truth
fobs = F.select(F.resolution_filter_selection(d_max=30, d_min=2.125))
r,c = utils.compute_r_factor( ftruth, fobs, verbose=False, is_flex=True, sort_flex=True)


ftruth2 = utils.open_flex(args.referenceForCCano)  # for comparing to CC ano
r2,c2 = utils.compute_r_factor( ftruth2, fobs, verbose=False, is_flex=True, sort_flex=True)

print("Rfactor with ground truth: %.2f %%" % (r*100))
print("CCano: %.2f %%" % (c2*100))
