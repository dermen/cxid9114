
from dxtbx.model.beam import BeamFactory
from dxtbx.model.detector import DetectorFactory

detDict={'hierarchy': {'children': [{'panel': 0}],
  'fast_axis': (1.0, 0.0, 0.0),
  'gain': 1.0,
  'identifier': '',
  'image_size': (0, 0),
  'mask': [],
  'material': '',
  'mu': 0.0,
  'name': '',
  'origin': (0.0, 0.0, 0.0),
  'pedestal': 0.0,
  'pixel_size': (0.0, 0.0),
  'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
  'raw_image_offset': (0, 0),
  'slow_axis': (0.0, 1.0, 0.0),
  'thickness': 0.0,
  'trusted_range': (0.0, 0.0),
  'type': ''},
 'panels': [{'fast_axis': (1.0, 0.0, 0.0),
   'gain': 1.0,
   'identifier': '',
   'image_size': (1800, 1800),
   'mask': [],
   'material': '',
   'mu': 0.0,
   'name': 'Panel',
   'origin': (-99.165, 99.05499999999999, -125.0),
   'pedestal': 0.0,
   'pixel_size': (0.11, 0.11),
   'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
   'raw_image_offset': (0, 0),
   'slow_axis': (0.0, -1.0, 0.0),
   'thickness': 0.0,
   'trusted_range': (-1e2, 1e7), #-29 65505.0),
   'type': 'SENSOR_CCD'}]}

beamDict ={'direction': (0.0, 0.0, 1.0),
 'divergence': 0.0,
 'flux': 0.0,
 'polarization_fraction': 0.999,
 'polarization_normal': (0.0, 1.0, 0.0),
 'sigma_divergence': 0.0,
 'transmission': 1.0,
 'wavelength': 1.37095}

BEAM = BeamFactory.from_dict(beamDict)
DET = DetectorFactory.from_dict(detDict)

