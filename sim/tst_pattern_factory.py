

def main():
    from cxid9114.sim import sim_utils
    from dxtbx.model.crystal import CrystalFactory
    from dxtbx_model_ext import flex_Beam
    from dxtbx.model.detector import DetectorFactory
    from dxtbx.model.beam import BeamFactory
    from simtbx.nanoBragg.tst_nanoBragg_basic import fcalc_from_pdb

    import numpy as np
    from cxid9114.parameters import ENERGY_CONV

    energies = np.arange(8920, 8930)
    fluxes = np.ones(len(energies)) * 5e11

    patt_args = {"Ncells_abc": (20, 20, 20),
                 "profile": "square",
                 "verbose": 0}

    beam_descr = {'direction': (0.0, 0.0, 1.0),
                 'divergence': 0.0,
                 'flux': 5e11,
                 'polarization_fraction': 1.,
                 'polarization_normal': (0.0, 1.0, 0.0),
                 'sigma_divergence': 0.0,
                 'transmission': 1.0,
                 'wavelength': ENERGY_CONV/energies[0]}

    cryst_descr = {'__id__': 'crystal',
                   'real_space_a': (79, 0, 0),
                   'real_space_b': (0, 79, 0),
                   'real_space_c': (0, 0, 38),
                   'space_group_hall_symbol': '-P 4 2'}

    det_descr = {'panels':
                   [{'fast_axis': (-1.0, 0.0, 0.0),
                     'gain': 1.0,
                     'identifier': '',
                     'image_size': (196, 196),
                     'mask': [],
                     'material': '',
                     'mu': 0.0,
                     'name': 'Panel',
                     'origin': (19.6, -19.6, -550),
                     'pedestal': 0.0,
                     'pixel_size': (0.1, 0.1),
                     'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
                     'raw_image_offset': (0, 0),
                     'slow_axis': (0.0, 1.0, 0.0),
                     'thickness': 0.0,
                     'trusted_range': (0.0, 65536.0),
                     'type': ''}]}

    DET = DetectorFactory.from_dict(det_descr)
    BEAM = BeamFactory.from_dict(beam_descr)

    crystal = CrystalFactory.from_dict(cryst_descr)
    Patt = sim_utils.PatternFactory(crystal=crystal, detector=DET, beam=BEAM,
                                    **patt_args)


    img = None
    Fens = []
    xrbeams = flex_Beam()
    for fl, en in zip(fluxes, energies):
        wave = ENERGY_CONV / en
        F = fcalc_from_pdb(resolution=4, algorithm="fft", wavelength=wave)
        Patt.primer(crystal, energy=en, flux=fl, F=F)
        if img is None:
            img = Patt.sim_rois(reset=True)  # defaults to full detector
        else:
            img += Patt.sim_rois(reset=True)

        Fens.append(F)

        xrb = BeamFactory.from_dict(beam_descr)
        xrb.set_wavelength(wave * 1e-10)  # need to fix the necessity to do this..
        xrb.set_flux(fl)
        xrb.set_direction(BEAM.get_direction())

        xrbeams.append(xrb)

    #import pylab as plt
    #def plot_img(ax,img):
    #    m = img[img >0].mean()
    #    s = img[img > 0].std()
    #    vmax = m+5*s
    #    vmin = 0
    #    ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gnuplot')

    print ("\n\n\n")
    print("<><><><><><><><><><>")
    print("NEXT TRIAL")
    print("<><><><><><><><><><>")
    #

    patt2 = sim_utils.PatternFactory(
        crystal=crystal, detector=DET,
        beam=xrbeams, **patt_args)

    patt2.prime_multi_Fhkl(multisource_Fhkl=Fens)
    img2 = patt2.sim_rois(reset=True)

    #plt.figure()
    #ax1 = plt.gca()
    #plt.figure()
    #ax2 = plt.gca()
    #plot_img(ax1, img)
    #plot_img(ax2, img2)
    #plt.show()

    assert(np.allclose(img, img2))


if __name__=="__main__":
    main()
    print("OK")
