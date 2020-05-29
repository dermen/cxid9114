
from cxid9114.sim import sim_utils
from dxtbx.model import ExperimentList
from cctbx import sgtbx,miller
from cctbx.crystal import symmetry
from dials.array_family import flex
import numpy as np
from cxid9114.parameters import ENERGY_CONV

defaultF = 1e3
mos_spread = 0
mos_doms = 1
Ncells_abc = 7, 7, 7
profile = "gauss"
beamsize = 0.1
exposure_s = 1
total_flux = 1e12
xtal_size = 0.0005


exper_name = "/Users/dermen/pinks/derek/refined_varying/stills2.expt"
El = ExperimentList.from_file(exper_name)
E = El[0]
crystal = E.crystal
DET = E.detector
BEAM = E.beam

symbol = "P212121"
sgi = sgtbx.space_group_info(symbol)
# TODO: allow override of ucell
symm = symmetry(unit_cell=crystal.get_unit_cell(), space_group_info=sgi)
miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
# NOTE does build_miller_set automatically expand to p1 ? Does it obey systematic absences ?
# Note how to handle sys absences here ?
Famp = flex.double(np.ones(len(miller_set.indices())) * defaultF)
mil_ar = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()

pinkstride = 50
wavelengths, weights = np.loadtxt("/Users/dermen/pinks/e080_2.lam", float,delimiter=',', skiprows=1).T
wavelengths = wavelengths[::pinkstride]
weights = weights[::pinkstride]

energies = ENERGY_CONV/wavelengths
FLUXES = weights / weights.sum() * total_flux

FF = [mil_ar] + [None]*(len(FLUXES)-1)
#from IPython import embed
#embed()
CUDA = False
device_Id = 0
show_params = False
simsAB = sim_utils.sim_colors(
    crystal, DET, BEAM, FF,
    energies,
    FLUXES, pids=None, profile=profile, cuda=CUDA, oversample=1,
    Ncells_abc=Ncells_abc, mos_dom=1, mos_spread=0,
    master_scale=1,
    exposure_s=exposure_s, beamsize_mm=beamsize, device_Id=device_Id,
    show_params=show_params, accumulate=False, crystal_size_mm=xtal_size)

