
import inspect
import numpy as np
import pylab as plt
import h5py
from scipy.interpolate import interp1d

import dxtbx
import cctbx
import dials.array_family.flex as flex
from scitbx.matrix import col
import simtbx.nanoBragg
nanoBragg = simtbx.nanoBragg.nanoBragg
shapetype = simtbx.nanoBragg.shapetype
convention = simtbx.nanoBragg.convention

from cxid9114 import utils
from cxid9114 import parameters


def mosaic_blocks(mos_spread_deg, mos_domains,
                  twister_seed=0, random_seed=1234):
    """
    Code from LS49 for adjusting mosaicity of simulation
    :param mos_spread_deg: spread in degrees
    :param mos_domains: number of mosaic domains
    :param twister_seed: default from ls49 code
    :param random_seed: default from ls49 code
    :return:
    """
    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=twister_seed)
    scitbx.random.set_random_seed(random_seed)
    rand_norm = scitbx.random.normal_distribution(mean=0,
                                                  sigma=mos_spread_deg*np.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(mos_domains)
    for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(m, deg=False) )
    return UMAT_nm


def compare_sims(SIM1, SIM2):
    """
    prints nanobragg params when they differ in SIM1 and SIM2 (useful for debug)
    :param SIM1: nanobragg instance
    :param SIM2: nanobragg instance
    :return:
    """
    isprop = lambda x: isinstance(x, property)
    bad_params = ('Fbg_vs_stol', 'Fhkl_tuple', 'amplitudes', 'indices', 'raw_pixels',
                  'xray_source_XYZ', 'xray_source_intensity_fraction', 'xray_beams',
                  'xray_source_wavelengths_A', 'unit_cell_tuple', 'progress_pixel')
    params = [name
              for (name, value) in inspect.getmembers(nanoBragg, isprop)
              if name not in bad_params]

    print "Did not try to get these parameters:"
    print bad_params

    failed = []
    for p in params:
        try:
            param_value1 = getattr(SIM1, p)
            param_value2 = getattr(SIM2, p)
            if isinstance(param_value1, tuple):
                params_are_equal = np.allclose( param_value1, param_value2)
            else:
                params_are_equal = param_value1 == param_value2
            if not params_are_equal:
                print p, param_value1
                print p, param_value2
                print
        except ValueError:
            failed.append(p)

    print "Failed to get these parameters:"
    print failed

def print_parameters(SIM):
    """
    prints nanobragg params
    :param SIM: nanobragg instance
    :return:
    """
    isprop = lambda x: isinstance(x, property)
    bad_params = ('Fbg_vs_stol', 'Fhkl_tuple', 'amplitudes', 'indices', 'raw_pixels',
                  'xray_source_XYZ', 'xray_source_intensity_fraction', 'xray_beams',
                  'xray_source_wavelengths_A', 'unit_cell_tuple', 'progress_pixel')
    params = [name
              for (name, value) in inspect.getmembers(nanoBragg, isprop)
              if name not in bad_params]

    print "Did not try to get these parameters:"
    print bad_params

    failed = []
    for p in params:
        try:
            param_value = getattr(SIM, p)
            print p, param_value
            print
        except ValueError:
            failed.append(p)

    print "Failed to get these parameters:"
    print failed


def Amatrix_dials2nanoBragg(crystal):
    """
    returns the A matrix from a cctbx crystal object
    in nanoBragg frormat
    :param crystal: cctbx crystal
    :return: Amatrix as a tuple
    """
    Amatrix = tuple(np.array(crystal.get_A()).reshape((3, 3)).T.ravel())
    return Amatrix


class PatternFactory:
    def __init__(self, crystal=None, detector=None, beam=None,
                 Ncells_abc=(10,10,10), Gauss=False, oversample=0, panel_id=0,
                 recenter=True, verbose=10, profile=None, device_Id=None,
                 beamsize_mm=None, exposure_s=None, flux=None):
        """
        :param crystal:  dials crystal model
        :param detector:  dials detector model
        :param beam: dials beam model
        """
        self.beam = beam
        self.detector = detector
        self.panel_id = panel_id
        if crystal is None:
            crystal = utils.open_flex(cryst_f)
        if self.detector is None:
            self.detector = utils.open_flex(det_f)
        if self.beam is None:
            self.beam = utils.open_flex(beam_f)

        self.SIM2 = nanoBragg(self.detector, self.beam, verbose=verbose, panel_id=panel_id)
        if oversample > 0:
            self.SIM2.oversample = oversample
        self.SIM2.polarization = 1  # polarization fraction ?
        self.SIM2.Ncells_abc = Ncells_abc  # important to set this First!
        self.SIM2.F000 = 0  # should be number of electrons ?
        self.SIM2.default_F = 0
        self.SIM2.Amatrix = Amatrix_dials2nanoBragg(crystal)  # sets the unit cell
        if Gauss:
            self.SIM2.xtal_shape = shapetype.Gauss
        else:
            self.SIM2.xtal_shape = shapetype.Tophat
        
        if profile is not None:  # override above
            if profile == "gauss":
                self.SIM2.xtal_shape = shapetype.Gauss
            elif profile == "tophat":
                self.SIM2.xtal_shape = shapetype.Tophat
            elif profile == "round":
                self.SIM2.xtal_shape = shapetype.Round
            elif profile == "square":
                self.SIM2.xtal_shape = shapetype.Square
       
        if device_Id is not None: 
            self.SIM2.device_Id=device_Id
            #self.SIM2.timelog=False
        self.SIM2.progress_meter = False
        if flux is not None:
            self.SIM2.flux = flux
        else:
            self.SIM2.flux = 1e14
        if beamsize_mm is not None:
            self.SIM2.beamsize_mm = beamsize_mm
        else:
            self.SIM2.beamsize_mm = 0.004

        self.SIM2.interpolate = 0
        self.SIM2.progress_meter = False
        self.SIM2.verbose = verbose
        self.SIM2.seed = 9012
        self.FULL_ROI = self.SIM2.region_of_interest

        if recenter:  # FIXME: I am not sure why this seems to be necessary to preserve geom
            #a = self.SIM2.beam_center_mm
            #print "Beam center was:",a
            self.SIM2.beam_center_mm = self.detector[panel_id].get_beam_centre(self.beam.get_s0())
            #print "Now, beam center is ", self.SIM2.beam_center_mm, "Why is this happening???"


    def adjust_mosaicity(self, mosaic_domains=None, mosaic_spread=None):
        if mosaic_domains is None:
            mosaic_domains = 2  # default
        if mosaic_spread is None:
            mosaic_spread = 0.1
        self.SIM2.mosaic_spread_deg = mosaic_spread  # from LS49
        self.SIM2.mosaic_domains = mosaic_domains  # from LS49
        self.SIM2.set_mosaic_blocks(mosaic_blocks(self.SIM2.mosaic_spread_deg,
                                                    self.SIM2.mosaic_domains))
    def adjust_divergence(self, div_tuple=(0,0)):
        self.SIM2.divergence_hv_mrad = div_tuple

    def adjust_dispersion(self, pct=0.):
        self.SIM2.dispersion_pct = pct

    def primer(self, crystal, energy, flux, F=None):
        self.SIM2.wavelength_A = parameters.ENERGY_CONV / energy
        self.SIM2.flux = flux
        # order of things is important here, Amatrix needs to be set
        # after Fhkl in current code!!
        if isinstance(F, cctbx.miller.array):
            self.SIM2.Fhkl = F.amplitudes()
        elif F is not None:
            self.SIM2.default_F = F
        if crystal is not None:
            self.SIM2.Amatrix = Amatrix_dials2nanoBragg(crystal)
        self.SIM2.raw_pixels *= 0
        self.SIM2.region_of_interest = self.FULL_ROI

    def sim_rois(self, rois, reset=True, cuda=False, omp=False, boost=1):
        for roi in rois:
            self.SIM2.region_of_interest = roi
            if cuda:
                print "Using CUDA"
                self.SIM2.add_nanoBragg_spots_cuda()
            elif omp:
                from boost.python import streambuf  # will deposit printout into dummy StringIO as side effect
                from six.moves import StringIO
                self.SIM2.add_nanoBragg_spots_nks(streambuf(StringIO()))
            else:
                self.SIM2.add_nanoBragg_spots()

        self.SIM2.raw_pixels = self.SIM2.raw_pixels*boost
    
        img = self.SIM2.raw_pixels.as_numpy_array()

        if reset:
            self.SIM2.raw_pixels *= 0
            self.SIM2.region_of_interest = self.FULL_ROI

        return img


def sim_colors(crystal, detector, beam, fcalcs, energies, fluxes, pids=None,
                   Gauss=False, oversample=0, Ncells_abc=(5,5,5),verbose=0,
                   div_tup=(0.,0.), disp_pct=0., mos_dom=2, mos_spread=0.15, profile=None,
                   roi_pp=None, counts_pp=None, cuda=False, omp=False, gimmie_Patt=False,
                   add_water=False, add_noise=False, boost=1, device_Id=0,
                   beamsize_mm=None, exposure_s=None, accumulate=False, only_water=False, add_spots=True):

    Npan = len(detector)
    Nchan = len(energies)

    if only_water:
        add_spots = False
        add_water = True
    else:
        add_spots = True
    
    # initialize output form
    panel_imgs = {}
    for i_en in range(Nchan):
        if accumulate:
            panel_imgs = [None]*Npan
        else:
            panel_imgs[i_en] = []

    if pids is None:
        pids = range(Npan)

    for ii, i_pan in enumerate(pids):
        PattF = PatternFactory(detector=detector,
                               crystal=crystal,
                               beam=beam,
                               panel_id=i_pan,
                               recenter=True,
                               Gauss=Gauss,
                               verbose=verbose,
                               Ncells_abc=Ncells_abc,
                               oversample=oversample, 
                               profile=profile, 
                               beamsize_mm=beamsize_mm,
                               exposure_s=exposure_s,
                               flux=np.sum(fluxes),
                               device_Id=device_Id)
        
        if not only_water:
            PattF.adjust_mosaicity(mos_dom, mos_spread)
        else:
            PattF.adjust_mosaicity(1,0)
       
        PattF.adjust_dispersion(disp_pct)
        PattF.adjust_divergence(div_tup)
        
        for i_en in range(Nchan):
            print i_en
            if fluxes[i_en] == 0:
                continue
            
            PattF.primer(crystal=crystal,
                         energy=energies[i_en],
                         F=fcalcs[i_en],
                         flux=fluxes[i_en])

            if roi_pp is None:
                color_img = PattF.sim_rois(rois=[PattF.FULL_ROI], 
                            reset=True, cuda=cuda, omp=omp,
                            boost=boost)
            else:
                color_img = PattF.sim_rois(rois=roi_pp[ii], reset=True, 
                            cuda=cuda, omp=omp,
                            boost=boost)
                where_finite = counts_pp[ii] > 0
                if np.any(where_finite):
                    color_img[where_finite] /= counts_pp[ii][where_finite]

            if accumulate:
                if panel_imgs[i_pan] is None:
                    panel_imgs[i_pan] = color_img
                else:
                    panel_imgs[i_pan] += color_img
            else:
                panel_imgs[i_en].append(color_img)

        PattF.SIM2.free_all()
    if gimmie_Patt:
        return panel_imgs, PattF
    else:
        #PattF.SIM2.free_all()
        #del PattF
        return panel_imgs


