
from __future__ import print_function
import inspect
import numpy as np
import time
import six
import sys
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import cctbx
import scitbx
import dials.array_family.flex as flex
from scitbx.matrix import col
import simtbx.nanoBragg
nanoBragg = simtbx.nanoBragg.nanoBragg
#nanoBragg = simtbx.nanoBragg.diffBragg
shapetype = simtbx.nanoBragg.shapetype
convention = simtbx.nanoBragg.convention

from cxid9114 import utils
from cxid9114 import parameters


def mosaic_blocks(mos_spread_deg, mos_domains,
                  twister_seed=None, random_seed=None):
    """
    Code from LS49 for adjusting mosaicity of simulation
    :param mos_spread_deg: spread in degrees
    :param mos_domains: number of mosaic domains
    :param twister_seed: default from ls49 code
    :param random_seed: default from ls49 code
    :return:
    """
    UMAT_nm = flex.mat3_double()
    if twister_seed is None:
        twister_seed = 777 
    if random_seed is None:
        random_seed = 777
    mersenne_twister = flex.mersenne_twister(seed=twister_seed)
    scitbx.random.set_random_seed(random_seed)
    rand_norm = scitbx.random.normal_distribution(mean=0,
                                                  sigma=mos_spread_deg*np.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(mos_domains)
    for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(m, deg=False))
        UMAT_nm.append(site.axis_and_angle_as_r3_rotation_matrix(-m, deg=False))  # NOTE: make symmetric dist
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

    print ("Did not try to get these parameters:")
    print (bad_params)

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
                print( p, param_value1)
                print( p, param_value2)
                print("\n")
        except ValueError:
            failed.append(p)

    print ("Failed to get these parameters:")
    print (failed)

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

    print("Did not try to get these parameters:")
    print(bad_params)

    failed = []
    for p in params:
        try:
            param_value = getattr(SIM, p)
            print(p, param_value)
            print("\n")
        except ValueError:
            failed.append(p)

    print("Failed to get these parameters:")
    print(failed)


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
                 beamsize_mm=.004, exposure_s=1, progress_meter=False,
                 crystal_size_mm=None, adc_offset=0, master_scale=None,amorphous_sample_thick_mm=0.005,
                 printout_pix=None):

        self.amorphous_sample_thick_mm = amorphous_sample_thick_mm
        self.beam = beam
        self._is_beam_a_flexBeam()
        self.detector = detector
        self.crystal = crystal
        self.master_scale = master_scale
        self.panel_id = int(panel_id)

        self.SIM2 = nanoBragg(self.detector, self.SIM_init_beam, verbose=verbose, panel_id=int(panel_id))
        if printout_pix is not None:
            self.SIM2.printout_pixel_fastslow = printout_pix
        if oversample > 0:
            self.SIM2.oversample = oversample
        self.SIM2.Ncells_abc = Ncells_abc  # important to set this First!
        self.SIM2.F000 = 0
        self.SIM2.default_F = 0
        self.SIM2.Amatrix = Amatrix_dials2nanoBragg(crystal)  # sets the unit cell
        self.crystal_size_mm = crystal_size_mm
        if crystal_size_mm is not None:
            if beamsize_mm > crystal_size_mm:
                self.crystal_volume = crystal_size_mm**3
            else:
                self.crystal_volume = self.crystal_size_mm*beamsize_mm*beamsize_mm
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
            if not isinstance(device_Id, int):
                try:
                    device_Id = int(device_Id)
                except:
                    raise ValueError("Device Id should be an integer!")
            self.SIM2.device_Id = int(device_Id)

        self.SIM2.adc_offset = adc_offset
        self.SIM2.beamsize_mm = beamsize_mm
        self.SIM2.exposure_s = exposure_s
        self.SIM2.interpolate = 0
        self.SIM2.progress_meter = progress_meter
        self.SIM2.verbose = verbose
        self.FULL_ROI = self.SIM2.region_of_interest

        a = self.SIM2.beam_center_mm
        if recenter:  # FIXME: I am not sure why this seems to be necessary to preserve geom
            self.SIM2.beam_center_mm = self.detector[int(panel_id)].get_beam_centre(self.SIM_init_beam.get_s0())
            
            #print ("Beam center was:",a)
            #print ("Now, beam center is ", self.SIM2.beam_center_mm, "Why is this happening???")
        else:
            self.SIM2.beam_center_mm = a

        if self.beam_is_flexBeam:
            self.SIM2.xray_beams = self.beam

    def _is_beam_a_flexBeam(self):
        try:
            _=self.beam.get_wavelength()
            self.beam_is_flexBeam = False
            self.SIM_init_beam = self.beam
        except AttributeError:
            self.beam_is_flexBeam = True
            # TODO insert checks to verify all directions in beam are same (only flux and energy can vary)
            from copy import deepcopy
            self.SIM_init_beam = deepcopy(self.beam[0])  # init sim with first beam
            # TODO: eliminate need to do this:
            self.SIM_init_beam.set_wavelength(self.SIM_init_beam.get_wavelength()*1e10)

    def adjust_mosaicity(self, mosaic_domains=25, mosaic_spread=0.01):
        self.SIM2.mosaic_spread_deg = mosaic_spread  # from LS49
        self.SIM2.mosaic_domains = mosaic_domains  # from LS49
        self.Umats = mosaic_blocks(self.SIM2.mosaic_spread_deg,
                                                    self.SIM2.mosaic_domains)
        self.SIM2.set_mosaic_blocks(self.Umats)

    def adjust_divergence(self, div_tuple=(0,0,0)):
        h, v, s = div_tuple
        self.SIM2.divergence_hv_mrad = (h,v)
        self.SIM2.divstep_hv_mrad =( s,s)

    def adjust_dispersion(self, pct=0.):
        self.SIM2.dispersion_pct = pct

    def primer(self, crystal, energy, flux, F=None):
        self.SIM2.wavelength_A = parameters.ENERGY_CONV / energy
        self.SIM2.flux = float(flux)
        # order of things is important here, Amatrix needs to be set
        #   after Fhkl in current code!!

        if self.crystal_size_mm is not None:
            try:
                self.mosaic_domain_volume = \
                    self.SIM2.xtal_size_mm[0]*self.SIM2.xtal_size_mm[1]*self.SIM2.xtal_size_mm[2]
                self.SIM2.spot_scale = self.crystal_volume / self.mosaic_domain_volume
                self.spot_scale = self.crystal_volume / self.mosaic_domain_volume
                if self.master_scale is not None:
                    self.spot_scale = self.master_scale
                    self.SIM2.spot_scale = self.master_scale
            except AttributeError:
                pass
        else:
            self.spot_scale = 1

        if isinstance(F, cctbx.miller.array):
            self.SIM2.Fhkl = F#.as_amplitude_array() #amplitudes()
        elif F is not None:
            self.SIM2.default_F = F
        
        if crystal is not None:
            self.SIM2.Amatrix = Amatrix_dials2nanoBragg(crystal)
        self.SIM2.raw_pixels *= 0
        self.SIM2.region_of_interest = self.FULL_ROI

    def prime_multi_Fhkl(self, multisource_Fhkl):

        assert self.beam_is_flexBeam
        assert len(multisource_Fhkl) == len(self.beam)
        self.SIM2.Multisource_Fhkl = multisource_Fhkl

    def sim_rois(self, rois=None, reset=True, cuda=False, omp=False,
                add_water=False, boost=1,
                add_spots=True, show_params=False):
        if show_params:
            self.SIM2.show_params()
            #if self.crystal_size_mm is not None:
            print("  Mosaic domain size mm = %.3g" % np.power(self.mosaic_domain_volume, 1/3.))
            print("  Spot scale = %.3g" % self.SIM2.spot_scale)

        if rois is None:
            rois = [self.FULL_ROI]
        if add_spots:
            for roi in rois:
                self.SIM2.region_of_interest = roi
                if cuda:
                    self.SIM2.add_nanoBragg_spots_cuda()
                elif omp:
                    from boost.python import streambuf  # will deposit printout into dummy StringIO as side effect
                    from six.moves import StringIO
                    self.SIM2.add_nanoBragg_spots_nks(streambuf(StringIO()))
                else:
                    self.SIM2.add_nanoBragg_spots()


            self.SIM2.raw_pixels = self.SIM2.raw_pixels*boost
        
        if add_water:  # add water for full panel, ignoring ROI
            water_scatter = flex.vec2_double(
                [(0, 2.57), (0.0365, 2.58), (0.07, 2.8), (0.12, 5), (0.162, 8), (0.2, 6.75), (0.18, 7.32),
                 (0.216, 6.75), (0.236, 6.5), (0.28, 4.5), (0.3, 4.3), (0.345, 4.36), (0.436, 3.77), (0.5, 3.17)])
            self.SIM2.Fbg_vs_stol = water_scatter
            self.SIM2.amorphous_density_gcm3 = 1
            self.SIM2.amorphous_molecular_weight_Da = 18
            self.SIM2.amorphous_sample_thick_mm = self.amorphous_sample_thick_mm 
            self.SIM2.region_of_interest = self.FULL_ROI
            self.SIM2.add_background()

        img = self.SIM2.raw_pixels.as_numpy_array()

        if reset:
            self.SIM2.raw_pixels *= 0
            self.SIM2.region_of_interest = self.FULL_ROI

        return img


def sim_colors(crystal, detector, beam, fcalcs, energies, fluxes, pids=None,
               Gauss=False, oversample=0, Ncells_abc=(5,5,5), verbose=0,
               div_tup=(0.,0.,0.), disp_pct=0., mos_dom=2, mos_spread=0.15, profile=None,
               roi_pp=None, counts_pp=None, cuda=False, omp=False, gimmie_Patt=False,
               add_water=False, boost=1, device_Id=0,
               beamsize_mm=0.001, exposure_s=1, accumulate=False, only_water=False, 
               add_spots=True, adc_offset=0, show_params=False, crystal_size_mm=None,
               amorphous_sample_thick_mm=0.005, free_all=True, master_scale=None,
               one_sf_array=False, printout_pix=None, time_panels=False, recenter=True):


    if not isinstance(energies, Iterable):
        raise ValueError("Energies needs to be an iterable")    
    if not isinstance(fluxes, Iterable):
        raise ValueError("Fluxes needs to be an iterable")    

    Npan = len(detector)
    Nchan = len(energies)
    if Nchan != len(fluxes):
        raise ValueError ("energies and fluxes need to be the same length")

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
        if time_panels:
            tstart = time.time()
        PattF = PatternFactory(detector=detector,
                               crystal=crystal,
                               beam=beam,
                               panel_id=i_pan,
                               recenter=recenter,
                               Gauss=Gauss,
                               verbose=verbose,
                               Ncells_abc=Ncells_abc,
                               oversample=oversample, 
                               profile=profile, 
                               beamsize_mm=beamsize_mm,
                               exposure_s=exposure_s,
                               device_Id=device_Id,
                               crystal_size_mm=crystal_size_mm,
                               adc_offset=adc_offset, 
                               master_scale=master_scale,
                               amorphous_sample_thick_mm=amorphous_sample_thick_mm,
                               printout_pix=printout_pix)
        if roi_pp is not None:
            assert( counts_pp is not None)
        
        if not only_water:
            PattF.adjust_mosaicity(mos_dom, mos_spread)
        else:
            PattF.adjust_mosaicity(1, 0)
       
        PattF.adjust_dispersion(disp_pct)
        #PattF.adjust_divergence(div_tup)
         
        en_count = 0 
        already_primed = False
        for i_en in range(Nchan):
            if fluxes[i_en] == 0:
                continue
            if one_sf_array:
                if not already_primed:
                    FCALC=fcalcs[0]
                    already_primed=True
                else:
                    FCALC=None
            else:
                FCALC=fcalcs[i_en]
            PattF.primer(crystal=crystal,
                         energy=energies[i_en],
                         F=FCALC,
                         flux=fluxes[i_en])
            
            if en_count==0 and ii==0 and show_params:
                show=True
            else:
                show=False
            if roi_pp is None:
                color_img = PattF.sim_rois(rois=[PattF.FULL_ROI], 
                            reset=True, cuda=cuda, omp=omp,
                            add_water=add_water, add_spots=add_spots,
                            boost=boost, show_params=show)
            else:
                color_img = PattF.sim_rois(rois=roi_pp[ii], reset=True, 
                            cuda=cuda, omp=omp,
                            add_water=add_water, add_spots=add_spots,
                            boost=boost, show_params=show)

                # if simulating ROI normalize for overlapping ROI (just in case)
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
            en_count += 1
       
        if free_all: 
            PattF.SIM2.free_all()
        if time_panels:
            tsim = time.time() - tstart
            if six.PY3:
                print("Panel %d (%d/%d) took %f sec" % (i_pan,ii+1, len(pids), tsim), flush=True) 
                #print("\rPanel %d (%d/%d) took %f sec" % (i_pan,ii+1, len(pids), tsim), end="", flush=True) 
            else:
                print("\rPanel %d (%d/%d) took %f sec" % (i_pan,ii+1, len(pids), tsim), end="") 
                sys.stdout.flush()
    if gimmie_Patt:
        return panel_imgs, PattF
    else:
        return panel_imgs


class microcrystal(object):
  # from LS49 but mod so works for beams bigger than the crystal
  def __init__(self, Deff_A, length_um, beam_diameter_um):
    # Deff_A is the effective domain size in Angstroms.
    # length_um is the effective path of the beam through the crystal in microns
    # beam_diameter_um is the effective (circular) beam diameter intersecting with the crystal in microns
    # assume a cubic crystal
    self.beam_area_um2 = np.pi* beam_diameter_um*beam_diameter_um / 4
    if beam_diameter_um > length_um:
      self.illuminated_volume_um3 = self.beam_area_um2*length_um
    else:
      self.illuminated_volume_um3 = length_um**3

    self.domain_volume_um3 = (4./3.)*np.pi*np.power( Deff_A / 2. / 1.E4, 3)
    self.domains_per_crystal = self.illuminated_volume_um3 / self.domain_volume_um3
    print("There are %d domains in the crystal"%self.domains_per_crystal)

    # assume circular domain projections and compute
    # number of domains per cross sectional area of beam focus
    Deff_um = Deff_A *1e-4
    self.domains_per_length = float(length_um) / Deff_um

  def number_of_cells(self, unit_cell):
    """if total then return total number of cells, else return number of cells
     normalized by cells per beam focus cross section"""
    cell_volume_um3 = unit_cell.volume()/np.power(1.E4,3)
    cells_per_domain = self.domain_volume_um3 / cell_volume_um3
    cube_root = np.power(cells_per_domain,1./3.)
   
    int_CR = round(cube_root,0)
    print("cells per domain", cells_per_domain,"%d x %d x %d"%(int_CR,int_CR,int_CR))
    return int(int_CR)

