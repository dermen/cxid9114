"""
Basic parameters for the experiment
"""
from scipy import constants

ENERGY_LOW = 8944
ENERGY_HIGH = 9034.7

ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt
WAVELEN_LOW = ENERGY_CONV / ENERGY_LOW
WAVELEN_HIGH = ENERGY_CONV / ENERGY_HIGH


