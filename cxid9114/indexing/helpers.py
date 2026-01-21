from cxid9114.prediction import prediction_utils
from cctbx import miller, sgtbx
from cctbx.crystal import symmetry
from dxtbx.model import Panel, Detector
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from cxid9114.parameters import ENERGY_CONV
from scipy import ndimage
import pylab as plt
from scitbx.array_family import flex


def get_flux_and_energy(beam=None, spec_file=None, total_flux=1e12, pinkstride=None):
    if spec_file is not None:
        FLUX, energies = load_spectra_file(spec_file, total_flux=1e12, pinkstride=pinkstride)
    else:
        assert beam is not None
        FLUX = [total_flux]
        energies = [ENERGY_CONV / Exper.beam.get_wavelength()]

    return FLUX, energies


def make_miller_array(symbol, unit_cell, defaultF=1000, d_min=1.5, d_max=999):
    sgi = sgtbx.space_group_info(symbol)
    # TODO: allow override of ucell
    symm = symmetry(unit_cell=unit_cell, space_group_info=sgi)
    miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
    # NOTE does build_miller_set automatically expand to p1 ? Does it obey systematic absences ?
    # Note how to handle sys absences here ?
    Famp = flex.double(np.ones(len(miller_set.indices())) * defaultF)
    mil_ar = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()
    return mil_ar

def load_spectra_file(spec_file, total_flux=1, pinkstride=1):
    wavelengths, weights = np.loadtxt(spec_file, float, delimiter=',', skiprows=1).T
    wavelengths = wavelengths[::pinkstride]
    weights = weights[::pinkstride]
    energies = ENERGY_CONV/wavelengths
    FLUXES = weights / weights.sum() * total_flux
    return FLUXES, energies


def make_background_pixel_mask(DET, refls_strong=None):
    # make mask of all strong spot pixels..
    n_panels = len(DET)
    nfast, nslow = DET[0].get_image_size()
    # make a mask that tells me True if I am a background pixel
    is_bg_pixel = np.ones((n_panels, nslow, nfast), bool)
    # group the refls by panel ID
    if refls_strong is None:
        return is_bg_pixel

    refls_strong_perpan = prediction_utils.refls_by_panelname(refls_strong)
    for panel_id in refls_strong_perpan:
        panel_id = int(panel_id)
        fast, slow = DET[panel_id].get_image_size()
        mask = prediction_utils.strong_spot_mask_dials(
            refls_strong_perpan[panel_id], (slow, fast),
            as_composite=True)
        # dilate the mask
        mask = binary_dilation(mask, iterations=args.dilate)
        is_bg_pixel[panel_id] = ~mask  # strong spots should not be background pixels

    return is_bg_pixel


#def boxes_from_panel_sims(sims, DET, BEAM, thresh, laue=False):
#
#    if laue:
#        patches
#
#def laue_boxes(panel_sims, edge=10, downsamp=1, threshold=1)
#    patches = []
#    bboxes = []
#    pids = []
#    for i_pan, panel_sim in enumerate(panel_sims):
#        # combined the different energy channels:
#        img = np.sum([panel_sim[i] for i in range(len(panel_sim))], 0)[0]
#        md_val = np.median(img[img > threshold])
#        mask = img > md_val
#        lab, nlab = ndimage.label(mask)
#        peaks = []
#        for i in range(1, nlab):
#            tot = np.sum(lab==i)
#            if tot > 1:
#                print(i)
#                peaks.append(i)
#
#        centroids = ndimage.center_of_mass(img, labels=lab, index=peaks)
#        y,x = zip(*centroids)
#
#        for i,j in zip(x,y):
#            i = i*downsamp-edge
#            j = j*downsamp-edge
#            R = Rectangle(xy=(i, j), width=2*edge, height=2*edge, fc='none', ec='w')
#            bbox = i,i+2*edge, j, j+2*edge
#            bboxes.append(bbox)
#            patches.append(R)
#            pids.append(pid)
#    return bboxes, patches, pids 

def downsample_detector(det, downsamp=1):
    newD = Detector()
    for panel in det:
        fast = panel.get_fast_axis()
        slow = panel.get_slow_axis()
        orig = panel.get_origin()
        panel_dict = panel.to_dict()
        panel_dict['fast'] = fast
        panel_dict['slow'] = slow
        panel_dict['origin'] = orig
        fast_dim, slow_dim = panel.get_image_size()
        panel_dict['image_size'] =  int(fast_dim/float(downsamp)), int(slow_dim/float(downsamp))
        pixsize = panel.get_pixel_size()[0]
        panel_dict['pixel_size'] =  pixsize*downsamp, pixsize*downsamp
        newpanel = Panel.from_dict(panel_dict)
        newD.add_panel(newpanel)	
    
    return newD

