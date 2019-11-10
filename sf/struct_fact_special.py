import numpy as np
import h5py
from scipy.interpolate import interp1d

from scitbx.array_family import flex
import os
from iotbx import pdb
from cctbx import sgtbx, crystal, miller
from cctbx.array_family import flex

from cxid9114.parameters import ENERGY_CONV
from cctbx.eltbx import henke, sasaki


class Yb_scatter:
    """special class for reporting Hendrickson paper fp and fdp"""

    def __init__(self, input_file=None):
        if input_file is None:
            input_file = "fp_fdp.npz"
        self.input_file = input_file

        if input_file is not None:
            self._load()

        self.hen_tbl = henke.table("Yb")
        self.sas_tbl = sasaki.table("Yb")

    def _load(self):
        self.data = np.load(self.input_file)
        self.ev_range = self.data["ev_range"]
        self.fp_raw = self.data["fp"]
        self.fdp_raw = self.data["fdp"]
        self.fp_I = interp1d(self.ev_range, self.fp_raw, kind='linear')
        self.fdp_I = interp1d(self.ev_range, self.fdp_raw, kind='linear')
        self.loaded = True

    def fp(self, wavelen_A):
        if not self.loaded:
            return None
        ev = ENERGY_CONV / wavelen_A
        return self.fp_I(ev)[()]

    def fdp(self, wavelen_A):
        if not self.loaded:
            return None
        ev = ENERGY_CONV / wavelen_A
        return self.fdp_I(ev)[()]

    def fp_tbl(self, wavelen_A=None, ev=None, henke=True):
        if wavelen_A is not None:
            ev = ENERGY_CONV / wavelen_A
        kev = ev * 1e-3
        if henke:
            factor = self.hen_tbl.at_kev(kev)
        else:
            factor = self.sas_tbl.at_kev(kev)
        return factor.fp()

    def fdp_tbl(self, wavelen_A=None, ev=None, henke=True):
        if wavelen_A is not None:
            ev = ENERGY_CONV / wavelen_A
        kev = ev * 1e-3
        if henke:
            factor = self.hen_tbl.at_kev(kev)
        else:
            factor = self.sas_tbl.at_kev(kev)
        return factor.fdp()


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
    import os
    sf_path = os.path.dirname(__file__)
    spec_file = os.path.join(sf_path, "../spec/realspec.h5")
    h5 = h5py.File(spec_file, "r")
    assert ("energy_bins" in h5.keys())
    en_chans = h5["energy_bins"][()]  # energy channels from spectrometer

    output_name = os.path.join(sf_path, "realspec_sfall.h5")  # output file name
    pdb_name = os.path.join(sf_path, '003_s0_mark0_001.pdb')  # refined pdb from sad data
    yb_scatter_name = os.path.join(sf_path, "scanned_fp_fdp.npz")  # high res fdp scan and corresponding calculated fp

    Fout = []
    indices_prev = None  # for sanity check on miller arrays
    for i_en, en in enumerate(en_chans):
        if i_en % 10 == 0:
            print ("Computing sf for energy channel %d / %d " % (i_en, len(en_chans)))
        wave = ENERGY_CONV / en
        F = sfgen(wave,
                  pdb_name,
                  algo='fft',
                  dmin=1.5,
                  ano_flag=True,
                  yb_scatter_name=yb_scatter_name)
        Fout.append(F.data().as_numpy_array())  # store structure factors in hdf5 format, needs numpy conversion

        # put in a sanity check on indices (same for all wavelengths ?)  
        if indices_prev is None:
            indices_prev = F.indices()
            continue
        assert (all(i == j for i, j in zip(F.indices(), indices_prev)))
        indices_prev = F.indices()

    hkl = F.indices()  # at this point, should be same hkl list for all energy channels
    hkl = np.vstack([hkl[i] for i in range(len(hkl))])
    with h5py.File(output_name, "w") as h5_out:
        h5_out.create_dataset("indices", data=hkl, dtype=np.int, compression="lzf")
        h5_out.create_dataset("data", data=np.vstack(Fout), compression="lzf")
        h5_out.create_dataset("energies", data=en_chans, compression="lzf")
        h5_out.create_dataset("ucell_tuple", data=F.unit_cell().parameters(), compression="lzf")
        h5_out.create_dataset("hall_symbol", data=F.space_group_info().type().hall_symbol())


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
    ucell_param = tuple(f["ucell_tuple"][()])
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


def load_4bs7_sf():
    from iotbx.reflection_file_reader import any_reflection_file
    import os
    from cctbx.array_family import flex
    from cctbx import miller
    sf_path = os.path.dirname(__file__)
    sf_file = os.path.join(sf_path, "4bs7-sf.cif")

    f = any_reflection_file(file_name=sf_file)
    reader = f.file_content()
    if reader is None:
        raise ValueError("Be sure to install git lfs and pull in the actual file with 4bs7-sf.cif")
    Fhkl = reader.build_miller_arrays()["r4bs7sf"]['_refln.F_meas_au_1']
    Fanom = Fhkl.generate_bijvoet_mates()
    Fdiff = reader.build_miller_arrays()['r4bs7sf']['_refln.pdbx_anom_difference_1']
    diff = {h: [val, sig] for h, val, sig in zip(Fdiff.indices(), Fdiff.data(), Fdiff.sigmas())}
    vals_anom = {h: [val, sig] for h, val, sig in zip(Fanom.indices(), Fanom.data(), Fanom.sigmas())}
    for H in diff:
        H_neg = -H[0], -H[1], -H[2]
        # compute Fhkl_minus
        val_low = vals_anom[H_neg][0] - .5*diff[H][0]

        vals_anom[H_neg][0] = val_low

        val_high = vals_anom[H_neg][0] + .5*diff[H][0]
        vals_anom[H][0] = val_high
        if val_low < 0:
            #res =
            offset = abs(val_low)
            vals_anom[H_neg][0] = val_low + offset*2
            vals_anom[H][0] = val_high + offset*2
        # propagate the error
        vals_anom[H_neg][1] = np.sqrt(vals_anom[H_neg][1]**2 + diff[H][1]**2)
        vals_anom[H][1] = np.sqrt(vals_anom[H_neg][1] ** 2 + diff[H][1] ** 2)

    hout = tuple(vals_anom.keys())
    mil_idx = flex.miller_index(hout)
    Symm = Fhkl.crystal_symmetry()
    mil_set = miller.set(crystal_symmetry=Symm, indices=mil_idx, anomalous_flag=True)
    Fdata = flex.double([vals_anom[h][0] for h in hout])
    Fsigmas = flex.double([vals_anom[h][1] for h in hout])
    Fhkl_anom = miller.array(mil_set, data=Fdata, sigmas=Fsigmas).set_observation_type_xray_amplitude()

    return Fhkl_anom


def karle_hendrickson_unknowns(d_min=1.5):
    sf_path = os.path.dirname(__file__)

    pdbname = os.path.join(sf_path, '003_s0_mark0_001.pdb')  # refined pdb from sad data
    pdbin = pdb.input(pdbname)
    xr = pdbin.xray_structure_simple()
    sc = xr.scatterers()
    sym = [s.element_symbol() for s in sc]

    print "Computing F total"
    Ft = xr.structure_factors(d_min=d_min,
                               algorithm='fft',
                               anomalous_flag=True).f_calc()

    yb_pos = [i for i, s in enumerate(sym) if s == 'Yb']
    yb_sel = flex.bool(len(sc), False)
    for pos in yb_pos:
        yb_sel[pos] = True
    yb_xr = xr.select(yb_sel)
    yb_sc = yb_xr.scatterers()

    print "Computing F heavy"
    Fh = yb_xr.structure_factors(d_min=d_min,
                               algorithm='fft',
                               anomalous_flag=True).f_calc()

    Ft_complex = Ft.data().as_numpy_array()
    Fh_complex = Fh.data().as_numpy_array()
    Ft_amp = np.abs(Ft_complex)
    Fh_amp = np.abs(Fh_complex)
    Fidx = [h for h in Ft.indices()]  # same
    assert Fidx == [h for h in Fh.indices()]
    Fidx = np.array(Fidx)
    positive_hand = np.all(Fidx >= 0, axis=1)
    phase_diff = np.angle(Ft_complex) - np.angle(Fh_complex)
    alphas = np.zeros_like(Ft_amp)
    for i, H in enumerate(Fidx):
        alpha = phase_diff[i]
        if not positive_hand[i]:
            alpha = -alpha
        alphas[i] = alpha
    hand = np.ones(len(Ft_amp))
    hand[~positive_hand] = -1

    return {"Ft": Ft_amp, "Fidx": Fidx, "Fh": Fh_amp, "alpha": alphas, "hand": hand}


if __name__ == "__main__":
    main()
    #out = karle_hendrickson_unknowns()
