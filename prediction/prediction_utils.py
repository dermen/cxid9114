

import numpy as np
from dials.array_family import flex
from scitbx.matrix import sqr


def refls_to_q(refls, detector, beam, update_table=False):

    orig_vecs = {}
    fs_vecs = {}
    ss_vecs = {}
    u_pids = set([r['panel'] for r in refls])
    for pid in u_pids:
        orig_vecs[pid] = np.array(detector[pid].get_origin())
        fs_vecs[pid] = np.array(detector[pid].get_fast_axis())
        ss_vecs[pid] = np.array(detector[pid].get_slow_axis())

    s1_vecs = []
    q_vecs = []
    for r in refls:
        pid = r['panel']
        i_fs, i_ss, _ = r['xyzobs.px.value']
        panel = detector[pid]
        orig = orig_vecs[pid] #panel.get_origin()
        fs = fs_vecs[pid] #panel.get_fast_axis()
        ss = ss_vecs[pid] #panel.get_slow_axis()

        fs_pixsize, ss_pixsize = panel.get_pixel_size()
        s1 = orig + i_fs*fs*fs_pixsize + i_ss*ss*ss_pixsize  # scattering vector
        s1 = s1 / np.linalg.norm(s1) / beam.get_wavelength()
        s1_vecs.append(s1)
        q_vecs.append(s1-beam.get_s0())

    if update_table:
        refls['s1'] = flex.vec3_double(tuple(map(tuple,s1_vecs)))
        refls['rlp'] = flex.vec3_double(tuple(map(tuple,q_vecs)))

    return np.vstack(q_vecs)


def refls_to_hkl(refls, detector, beam, crystal,
                 update_table=False, returnQ=False):
    """
    convert pixel panel reflections to miller index data

    :param refls:  reflecton table for a panel or a tuple of (x,y)
    :param detector:  dxtbx detector model
    :param beam:  dxtbx beam model
    :param crystal: dxtbx crystal model
    :param update_table: whether to update the refltable
    :param returnQ: whether to return intermediately computed q vectors
    :return: if as_numpy two Nx3 numpy arrays are returned
        (one for fractional and one for whole HKL)
        else dictionary of hkl_i (nearest) and hkl (fractional)
    """
    if 'rlp' not in list(refls.keys()):
        q_vecs = refls_to_q(refls, detector, beam, update_table=update_table)
    else:
        q_vecs = np.vstack([r['rlp'] for r in refls])
    Ai = sqr(crystal.get_A()).inverse()
    Ai = Ai.as_numpy_array()
    HKL = np.dot( Ai, q_vecs.T)
    HKLi = map( lambda h: np.ceil(h-0.5).astype(int), HKL)
    if update_table:
        refls['miller_index'] = flex.miller_index(len(refls),(0,0,0))
        mil_idx = flex.vec3_int(tuple(map(tuple, np.vstack(HKLi).T)))
        for i in range(len(refls)):
            refls['miller_index'][i] = mil_idx[i]
    if returnQ:
        return np.vstack(HKL).T, np.vstack(HKLi).T, q_vecs
    else:
        return np.vstack(HKL).T, np.vstack(HKLi).T


def xyz_from_refl(refl, key="xyzobs.px.value"):
    """returns the xyz of the pixels by default in weird (xpix, ypix, zmm) format"""
    x,y,z = zip( * [refl[key][i] for i in range(len(refl))])
    return x,y,z



def get_prediction_boxes(refls_at_colors, detector, beams_of_colors, crystal,
                    twopi_conv=True, delta_q=0.015,
                    ret_patches=False, **kwargs):
    """
    :param refls_at_colors: List of reflection tables, one for each color in the experiment
    :param detector: dxtbx detector model
    :param beams_of_colors: one beam object per color
    :param crystal: crystal model
    :param twopi_conv: use two pi in q cal
    :param delta_q: widt of boxes in inverse angstroms (so boxes get bigger at higher scattering angles)
    :param ret_patches: return a patch collection to plot boxes on top of image in matplotlib
    :param kwargs: kwargs to define the patches, only matters is ret_patches is True
    :return:  bound boxes of the spots on the detector, alternatively also returns the matplotlib patches
    """
    if ret_patches:
        import matplotlib as mpl
        import pylab as plt

    color_data = {}
    color_data["Q"] = []
    color_data["H"] = []
    color_data["Hi"] = []
    color_data["x"] = []
    color_data["y"] = []
    color_data["Qmag"] = []
    detdist = detector[0].get_distance()
    pixsize = detector[0].get_pixel_size()[0]
    fs_dim, ss_dim = detector[0].get_image_size()
    for refls, beam in zip(refls_at_colors, beams_of_colors):

        H, Hi, Q = refls_to_hkl(
            refls, detector, beam, crystal,  returnQ=True)

        color_data["Q"].append(list(Q))
        color_data["H"].append(list(H))
        color_data["Hi"].append(list(map(tuple, Hi)))
        Qmag = np.linalg.norm(Q, axis=1)
        if twopi_conv:
            Qmag*=2*np.pi

        x, y, _ = xyz_from_refl(refls)
        color_data["x"].append(x)
        color_data["y"].append(y)
        color_data["Qmag"].append(Qmag)

    ave_wave = np.mean([beam.get_wavelength() for beam in beams_of_colors])
    all_indexed_Hi = [tuple(h) for hlist in color_data["Hi"] for h in hlist]
    unique_indexed_Hi = set(all_indexed_Hi)

    all_x, all_y, all_H = [], [], []
    patches = []
    bboxes = []
    for H in unique_indexed_Hi:
        x_com = 0
        y_com = 0
        Qmag = 0
        n_counts = 0
        for i_color in range(len(beams_of_colors)):
            in_color = H in color_data["Hi"][i_color]
            if not in_color:
                continue

            idx = color_data["Hi"][i_color].index(H)
            x_com += color_data["x"][i_color][idx] - 0.5
            y_com += color_data["y"][i_color][idx] - 0.5
            Qmag += color_data["Qmag"][i_color][idx]
            n_counts += 1
        Qmag = Qmag / n_counts
        all_x.append(x_com / n_counts)
        all_y.append(y_com / n_counts)

        x_com = x_com / n_counts
        y_com = y_com / n_counts

        rad1 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag-delta_q*.5)*ave_wave/4/np.pi))
        rad2 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag+delta_q*.5)*ave_wave/4/np.pi))
        delrad = rad2-rad1

        i1 = int(max(x_com - delrad/2., 0))
        i2 = int(min(x_com + delrad/2., fs_dim))
        j1 = int(max(y_com - delrad/2., 0))
        j2 = int(min(y_com + delrad/2., ss_dim))

        bboxes.append((i1,i2,j1,j2))   # i is fast scan, j is slow scan

        if ret_patches:
            R = plt.Rectangle(xy=(x_com-delrad/2., y_com-delrad/2.),
                              width=delrad,
                              height=delrad,
                              **kwargs)
            patches.append(R)

    unique_indexed_Hi = set(all_indexed_Hi)
    if ret_patches:
        patch_coll = mpl.collections.PatchCollection(patches,
                          match_original=True)
        return list(unique_indexed_Hi), bboxes, patch_coll
    else:
        return list(unique_indexed_Hi), bboxes


def refls_from_sims(panel_imgs, detector, beam, thresh=0, filter=None, panel_ids=None, **kwargs):
    """
    This class is for converting the centroids in the noiseless simtbx images
    to a multi panel reflection table

    :param panel_imgs: list or 3D array of detector panel simulations
    :param detector: dxtbx  detector model of a caspad
    :param beam:  dxtxb beam model
    :param thresh: threshol intensity for labeling centroids
    :param filter: optional filter to apply to images before
        labeling threshold, typically one of scipy.ndimage's filters
    :param pids: panel IDS , else assumes panel_imgs is same length as detector
    :param kwargs: kwargs to pass along to the optional filter
    :return: a reflection table of spot centroids
    """
    from dials.algorithms.spot_finding.factory import FilterRunner
    from dials.model.data import PixelListLabeller, PixelList
    from dials.algorithms.spot_finding.finder import PixelListToReflectionTable
    from cxid9114 import utils

    if panel_ids is None:
        panel_ids = np.arange(len(detector))
    pxlst_labs = []
    for i, pid in enumerate(panel_ids):
        plab = PixelListLabeller()
        img = panel_imgs[i]
        if filter is not None:
            mask = filter(img, **kwargs) > thresh
        else:
            mask = img > thresh
        img_sz = detector[pid].get_image_size()
        flex_img = flex.double(img)
        flex_img.reshape(flex.grid(img_sz))

        flex_mask = flex.bool(mask)
        flex_mask.resize(flex.grid(img_sz))
        pl = PixelList(0, flex.double(img), flex.bool(mask))
        plab.add(pl)

        pxlst_labs.append(plab)

    pixlst_to_reftbl = PixelListToReflectionTable(
        min_spot_size=1,
        max_spot_size=194 * 184,  # TODO: change this ?
        filter_spots=FilterRunner(),  # must use a dummie filter runner!
        write_hot_pixel_mask=False)

    #dblock = utils.datablock_from_numpyarrays(panel_imgs, detector, beam)
    #iset = dblock.extract_imagesets()[0]
    El = utils.explist_from_numpyarrays(panel_imgs, detector, beam)
    iset = El.imagesets()[0]
    refls = pixlst_to_reftbl(iset, pxlst_labs)[0]

    return refls
