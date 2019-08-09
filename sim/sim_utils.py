
import inspect
import numpy as np

import cctbx
import scitbx
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
                 beamsize_mm=.004, exposure_s=1, flux=1e12):

        self.beam = beam
        self.detector = detector
        self.crystal = crystal
        self.panel_id = panel_id

        self.SIM2 = nanoBragg(self.detector, self.beam, verbose=verbose, panel_id=panel_id)
        if oversample > 0:
            self.SIM2.oversample = oversample
        self.SIM2.polarization = 1  # polarization fraction ?
        self.SIM2.Ncells_abc = Ncells_abc  # important to set this First!
        self.SIM2.F000 = 0
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

        #self.SIM2.flux = flux
        self.SIM2.beamsize_mm = beamsize_mm
        self.SIM2.exposure_s = exposure_s
        self.SIM2.interpolate = 0
        self.SIM2.progress_meter = False
        self.SIM2.verbose = verbose
        self.FULL_ROI = self.SIM2.region_of_interest

        if recenter:  # FIXME: I am not sure why this seems to be necessary to preserve geom
            #a = self.SIM2.beam_center_mm
            #print "Beam center was:",a
            self.SIM2.beam_center_mm = self.detector[panel_id].get_beam_centre(self.beam.get_s0())
            #print "Now, beam center is ", self.SIM2.beam_center_mm, "Why is this happening???"


    def adjust_mosaicity(self, mosaic_domains=25, mosaic_spread=0.01):
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
        #   after Fhkl in current code!!
        if isinstance(F, cctbx.miller.array):
            self.SIM2.Fhkl = F.amplitudes()
        elif F is not None:
            self.SIM2.default_F = F
        if crystal is not None:
            self.SIM2.Amatrix = Amatrix_dials2nanoBragg(crystal)
        self.SIM2.raw_pixels *= 0
        self.SIM2.region_of_interest = self.FULL_ROI

    def sim_rois(self, rois, reset=True, cuda=False, omp=False,
                add_water=False, boost=1,
                add_spots=True):
        
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
            self.SIM2.amorphous_sample_thick_mm = 0.005  # typical GDVN jet thickness but could vary
            self.SIM2.amorphous_density_gcm3 = 1
            self.SIM2.amorphous_molecular_weight_Da = 18
            self.SIM2.region_of_interest = self.FULL_ROI
            self.SIM2.add_background()

        img = self.SIM2.raw_pixels.as_numpy_array()

        if reset:
            self.SIM2.raw_pixels *= 0
            self.SIM2.region_of_interest = self.FULL_ROI

        return img


def sim_colors(crystal, detector, beam, fcalcs, energies, fluxes, pids=None,
               Gauss=False, oversample=0, Ncells_abc=(5,5,5),verbose=0,
               div_tup=(0.,0.), disp_pct=0., mos_dom=2, mos_spread=0.15, profile=None,
               roi_pp=None, counts_pp=None, cuda=False, omp=False, gimmie_Patt=False,
               add_water=False, boost=1, device_Id=0,
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
                            add_water=add_water, add_spots=add_spots,
                            boost=boost)
            else:
                color_img = PattF.sim_rois(rois=roi_pp[ii], reset=True, 
                            cuda=cuda, omp=omp,
                            add_water=add_water, add_spots=add_spots,
                            boost=boost)

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

        PattF.SIM2.free_all()

    if gimmie_Patt:
        return panel_imgs, PattF
    else:
        return panel_imgs


class microcrystal(object):
  # from LS49 but mod so works for beams bigger than the crystal
  def __init__(self, Deff_A, length_um, beam_diameter_um):
    from libtbx import adopt_init_args
    adopt_init_args(self, locals())
    # Deff_A is the effective domain size in Angstroms.
    # length_um is the effective path of the beam through the crystal in microns
    # beam_diameter_um is the effective (circular) beam diameter intersecting with the crystal in microns
    # assume a cubic crystal
    if beam_diameter_um > length_um:
      self.illuminated_volume_um3 = math.pi * (beam_diameter_um/2.) * (beam_diameter_um/2.) * length_um
    else:
      self.illuminated_volume_um3 = length_um**3

    self.domain_volume_um3 = (4./3.)*math.pi*math.pow( Deff_A / 2. / 1.E4, 3)
    self.domains_per_crystal = self.illuminated_volume_um3 / self.domain_volume_um3
    print("There are %d domains in the crystal"%self.domains_per_crystal)
  def number_of_cells(self, unit_cell):
    cell_volume_um3 = unit_cell.volume()/math.pow(1.E4,3)
    cells_per_domain = self.domain_volume_um3 / cell_volume_um3
    cube_root = math.pow(cells_per_domain,1./3.)
    int_CR = round(cube_root,0)
    print("cells per domain",cells_per_domain,"%d x %d x %d"%(int_CR,int_CR,int_CR))
    return int(int_CR)
