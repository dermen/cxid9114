
import numpy as np
import h5py
from scipy.interpolate import interp1d

from iotbx import pdb
from cctbx.eltbx import henke
from cctbx import sgtbx, crystal, miller
from cctbx.array_family import flex

from cxid9114.parameters import ENERGY_CONV


class Yb_scatter:
    """special class for reporting Hendrickson paper fp and fdp"""

    def __init__(self, input_file=None):
        if input_file is None:
            input_file = "fp_fdp.npz"
        self.input_file = input_file
        
        self._load()

    def _load(self):
        self.data = np.load(self.input_file)
        self.ev_range = self.data["ev_range"]
        self.fp_raw = self.data["fp"]
        self.fdp_raw = self.data["fdp"]
        self.fp_I = interp1d(self.ev_range, self.fp_raw, kind='linear')
        self.fdp_I = interp1d(self.ev_range, self.fdp_raw, kind='linear')

    def fp(self, wavelen_A):
        ev = ENERGY_CONV/wavelen_A
        return self.fp_I(ev)[()]

    def fdp(self, wavelen_A):
        ev = ENERGY_CONV/wavelen_A
        return self.fdp_I(ev)[()]


def sfgen(wavelen_A, pdb_name, algo='fft', dmin=1.5, ano_flag=True, yb_scatter_name=None):
    """
    generate the structure factors from a pdb
    for use with LD91
    :param wavelen_A:
    :param pdb_name:
    :param algo:
    :param dmin:
    :param ano_flag:
    :param yb_scatter_name:
    :return:
    """
    pdb_in = pdb.input(pdb_name)
    xray_structure = pdb_in.xray_structure_simple()
    scatts = xray_structure.scatterers()

    Yb = Yb_scatter(input_file=yb_scatter_name)  # reads in values from the Hendrickson plot

    for sc in scatts:
        if sc.element_symbol() == "Yb":
            sc.fp = Yb.fp(wavelen_A)
            sc.fdp = Yb.fdp(wavelen_A)
        else:
            expected_henke = henke.table(
                sc.element_symbol()).at_angstrom(wavelen_A)
            sc.fp = expected_henke.fp()
            sc.fdp = expected_henke.fdp()

    fcalc = xray_structure.structure_factors(
        d_min=dmin,
        algorithm=algo,
        anomalous_flag=ano_flag)
    return fcalc.f_calc()


def main():
    en_chans = h5py.File("exper_spectra.h5", "r")["energy_bins"][()]  # energy channels from spectrometer
    output_name = "exper_spectra_sfall.h5"  # output file name
    pdb_name='003_s0_mark0_001.pdb'  # refined pdb from sad data
    yb_scatter_name = "scanned_fp_fdp.npz"  # calc fp from high-res fdp

    Fout = []
    for i_en, en in enumerate(en_chans):
        print i_en, len(en_chans)
        wave = ENERGY_CONV / en
        F = sfgen(wave,
                  pdb_name,
                  algo='fft',
                  dmin=1.5,
                  ano_flag=True,
                  yb_scatter_name=yb_scatter_name)
        Fout.append(F.data().as_numpy_array())
    hkl = F.indices()   # should be same hkl list for all channels
    hkl = np.vstack([hkl[i] for i in range(len(hkl))])
    with h5py.File(output_name, "w") as h5_out:
        h5_out.create_dataset("indices", data=hkl, dtype=np.int, compression="lzf")
        h5_out.create_dataset("data", data=np.vstack(Fout), compression="lzf")
        h5_out.create_dataset("energies", data=en_chans, compression="lzf")
        h5_out.create_dataset("ucell_tuple", data=F.unit_cell().parameters(), compression="lzf")
        h5_out.create_dataset("hall_symbol", data=F.space_group_info().type().hall_symbol() )


def load_sfall(fname): 
    """
    special script for loading the structure factor file generated in main()
    :param fname: file generated in the main method above.. 
    :return: mil_ar, energies
        mil_ar: dict of miller arrays (complex) 
        energies: array of xray energies in electron volts
        such that  mil_ar[0] is Fhkl at energy energies[0]
    """

    f = h5py.File(fname, "r")
    data = f["data"][()]
    indices = f["indices"][()]
    hall = f["hall_symbol"][()]
    ucell_param = f["ucell_tuple"][()]
    energies = f["energies"][()]
    sg = sgtbx.space_group(hall)
    Symm = crystal.symmetry(unit_cell=ucell_param, space_group=sg)
    indices_flex = tuple(map(tuple, indices))
    mil_idx = flex.miller_index(indices_flex)
    mil_set = miller.set(crystal_symmetry=Symm, indices=mil_idx, anomalous_flag=True)

    mil_ar = {}  # load a dict of "sfall at each energy"
    for i_chan, data_chan in enumerate(data):
        data_flex = flex.complex_double(np.ascontiguousarray(data_chan))
        mil_ar[i_chan] = miller.array(mil_set, data=data_flex)

    return mil_ar, energies


if __name__=="__main__":
    main()

