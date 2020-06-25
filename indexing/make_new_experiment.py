from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--pkl", type=str, required=True, help="path to pandas pickle")
parser.add_argument("--exp", type=str, required=True, help="path to combined experimnts (should have matching refl)")
parser.add_argument("--tag", type=str, required=True, help="output prefix for new experiment and refl files")
args = parser.parse_args()


from dxtbx.model import Experiment, ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
from copy import deepcopy
from IPython import embed
import pandas
from dials.array_family import flex

#fname = '/global/cfs/cdirs/lcls/dermen/d9114_sims/indexing_kaladin_2k/filtered.expt'
El = ExperimentListFactory.from_json_file(args.exp, check_format=False)
R = flex.reflection_table.from_file(args.exp.replace(".expt", ".refl"))

beams = El.beams()
crystals = El.crystals()
#df = pandas.read_pickle('results_df2_filt0_second.pkl')
#df2_filt0 = df.query("spot_scales < 0.03").query("9.6 < ncells < 10.4").query("38.38 < c < 38.42").query("79.08 < a < 79.12")
df2_filt0 = pandas.read_pickle(args.pkl)

isets = El.imagesets()
R2 = flex.reflection_table()
El2 = ExperimentList()
D = El.detectors()[0]
N = len(isets)
new_id = 0
has_master = False
if 'master_indices' in list(df2_filt0):
    has_master = True

test = df2_filt0.imgpaths.values[0]
try:
    test.decode()
    df2_filt0['imgpaths'] = df2_filt0.imgpaths.str.decode('utf8')
except AttributeError:
    pass


for i in range(N):
    iset = isets[i]
    path = iset.get_path(0)
    d = df2_filt0.query("imgpaths=='%s'" % path)
    if has_master:
        master_index = iset.indices()[0]
        d = d.query("master_indices==%d" % master_index)
    if len(d) != 1:
        continue
    A = d.Amats.values[0]
    #break
    C = deepcopy(crystals[i])
    #C.set_A(A)
    Ex = Experiment()
    Ex.crystal = C
    Ex.imageset = iset
    Ex.beam = beams[i]
    Ex.detector = D
    El2.append(Ex)

    Rsel = R.select(R['id']==i)
    nref = len(Rsel)
    Rsel['id'] = flex.int(nref, new_id)
    R2.extend(Rsel)
    new_id += 1
    print (new_id)

el_file = "%s.expt" % args.tag
R_file = "%s.refl" % args.tag
El2.as_file(el_file)
print("Saved experiment %s" % el_file )
R2.as_file(R_file)
print("Saved refls %s" % R_file)

