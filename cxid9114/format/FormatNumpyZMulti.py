from __future__ import absolute_import, division

import numpy as np

from dxtbx.format.Format import Format
from dxtbx.format.FormatStill import FormatStill
from dials.array_family import flex
import six


class FormatNumpyMultiZ(FormatStill, Format):
    """
    Class for reading D9114 simulated multipanel data
    """
    @staticmethod
    def understand(image_file):
        if six.PY3:
            return False
        try:
            img_handle = np.load(image_file)
            keys = img_handle.keys()
        except (IOError, AttributeError) as err:
            return False
        if "det" not in keys:
            return False
        if "beam" not in keys:
            return False
        if "img" not in keys:
            return False
        if len(img_handle["img"].shape)==2:
            return False
        return True

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatStill.__init__(self, image_file, **kwargs)
        self._handle = np.load(image_file)
        self._geometry_define()

    def _geometry_define(self):
        self._cctbx_detector = self._detector_factory.from_dict(self._handle["det"][()])
        self._cctbx_beam = self._beam_factory.from_dict(self._handle["beam"][()])

    def get_num_images(self):
        return 1

    def load_panel_imgs(self):
        self.panel_imgs = self._handle["img"]
        if not self.panel_imgs.dtype == np.float64:
            self.panel_imgs = self.panel_imgs.astype(np.float64)

    def get_raw_data(self, index=0):
        self.load_panel_imgs()
        return tuple([flex.double(I) for I in self.panel_imgs])

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_detector(self, index=None):
        return self._cctbx_detector

    def get_beam(self, index=None):
        return self._cctbx_beam


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(FormatNumpyMultiZ.understand(arg))
