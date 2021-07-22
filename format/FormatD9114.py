from __future__ import absolute_import, division

import numpy as np
import h5py
import json
from copy import deepcopy
import ast

from dxtbx.format.FormatHDF5 import FormatHDF5
from dials.array_family import flex
from dxtbx.format.FormatStill import FormatStill
from dxtbx.model import Beam, Spectrum


class FormatD9114(FormatHDF5, FormatStill):
    """
    Class for reading HDF5 files for arbitrary geometries
    focused on performance
    """
    @staticmethod
    def understand(image_file):
        try:
            img_handle = h5py.File(image_file, "r")
            keys = img_handle.keys()
        except (IOError, AttributeError) as err:
            return False
        if "images" not in keys:
            return False
        if "wavelengths" not in keys or "spectrum" not in keys:
            return False
        images = img_handle["images"]
        if "dxtbx_detector_string" not in images.attrs:
            return False
        if "dxtbx_beam_string" not in images.attrs:
            return False
        return True

    def _start(self):
        self._handle = h5py.File(self._image_file, "r")
        self._image_dset = self._handle["images"]
        self._geometry_define()
        self._energies = None
        self._weights = None
        self._central_wavelengths = None
        self._load_per_shot_spectra()
        self._ENERGY_CONV = 12398.419739640716

    def _geometry_define(self):
        det_str = self._image_dset.attrs["dxtbx_detector_string"]
        beam_str = self._image_dset.attrs["dxtbx_beam_string"]
        try:
            det_str = det_str.decode()
            beam_str = beam_str.decode()
        except AttributeError:
            pass
        det_dict = ast.literal_eval(det_str)
        beam_dict = ast.literal_eval(beam_str)
        self._cctbx_detector = self._detector_factory.from_dict(det_dict)
        self._cctbx_beam = self._beam_factory.from_dict(beam_dict)

    def _load_per_shot_spectra(self):
        self._energies = self._ENERGY_CONV/self._handle["wavelengths"][()]
        self._weights = self._handle["spectrum"][()]

    def get_num_images(self):
        return self._image_dset.shape[0]

    def get_raw_data(self, index=0):
        self.panels = self._image_dset[index]
        if self.panels.dtype == np.float64:
            flex_data = [flex.double(p) for p in self._image_dset[index]]
        else:
            flex_data = [flex.double(p.astype(np.float64)) for p in self._image_dset[index]]
        return tuple(flex_data)

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_detector(self, index=None):
        return self._cctbx_detector

    def _get_wavelength(self, index):
        assert index==0
        w = self._weights
        E = self._energies
        ave_E = (w*E).sum() / (w.sum())
        wavelength = self._ENERGY_CONV / ave_E 
        return wavelength

    def get_beam(self, index=0):
        beam = self._cctbx_beam
        wavelength = self._get_wavelength(index)
        beam = deepcopy(self._cctbx_beam)
        beam.set_wavelength(wavelength)
        return beam

    def get_spectrum(self, index=0):
        spectrum = Spectrum(self._energies, self._weights)
        return spectrum


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(FormatD9114.understand(arg))
