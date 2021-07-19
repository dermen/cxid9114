
import numpy as np
from dials.array_family import flex
from scitbx.matrix import sqr
from cxid9114.integrate import integrate_utils
from dxtbx.imageset import MemReader
from dxtbx.imageset import ImageSet, ImageSetData
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import ExperimentList

def refls_by_panelname(refls):
    Nrefl = len(refls)
    panel_names = np.array([refls["panel"][i] for i in range( Nrefl)])
    uniq_names = np.unique(panel_names)
    refls_by_panel = {name: refls.select(flex.bool(panel_names == name))
                      for name in uniq_names}
    return refls_by_panel


def refls_to_q(refls, detector, beam, update_table=False):

    orig_vecs = {}
    fs_vecs = {}
    ss_vecs = {}
    u_pids = set(refls['panel'])
    for pid in u_pids:
        orig_vecs[pid] = np.array(detector[pid].get_origin())
        fs_vecs[pid] = np.array(detector[pid].get_fast_axis())
        ss_vecs[pid] = np.array(detector[pid].get_slow_axis())

    s1_vecs = []
    q_vecs = []
    panels = refls["panel"]
    n_refls = len(refls)
    for i_r in range(n_refls):
        r = refls[i_r]
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
        refls['s1'] = flex.vec3_double(tuple(map(tuple, s1_vecs)))
        refls['rlp'] = flex.vec3_double(tuple(map(tuple, q_vecs)))

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
    HKLi = list(map( lambda h: np.ceil(h-0.5).astype(int), HKL))
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
                    ret_patches=False, ret_Pvals=False, data=None, refls_data=None,
                    bad_pixel_mask=None, gain=1, **kwargs):
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
        import pylab as plt
    if ret_Pvals:
        assert data is not None
        assert refls_data is not None
        color_integration_masks = {}
        for i_color, refls in enumerate(refls_at_colors):
            color_integration_masks[i_color] = \
                strong_spot_mask(refls, detector)
        allspotmask = strong_spot_mask(refls_data, detector)
        if bad_pixel_mask is None:
            bad_pixel_mask = np.ones_like(allspotmask, bool)
        integrated_Hi = []

    panel_masks = {}
    for pid in range(len(detector)):
        panel_masks[pid] = None

    for r in refls_at_colors:
        rpp = refls_by_panelname(r)
        for panel_id in rpp:
            fast, slow = detector[int(panel_id)].get_image_size()  # int casting necessary in Py3
            mask = strong_spot_mask_dials(rpp[panel_id], (slow, fast),
                                   as_composite=True)
            if panel_masks[panel_id] is None:
                panel_masks[panel_id] = mask
            else:
                panel_masks[panel_id] = np.logical_or(mask, panel_masks[panel_id])

    color_data = {}
    color_data["Q"] = []  # momentum transfer vector
    color_data["H"] = []  # miller index fractional
    color_data["Hi"] = []  # miller index
    color_data["x"] = []  # fast scan coordinate in panel
    color_data["y"] = []  # slow scane coordinate in panel
    color_data["Qmag"] = []  # momentum transfer magnitude
    color_data["panel"] = []  # panel id in detector model
    color_data["Pterms"] = []  # summed Bragg spot intensity
    color_data["strong_masks"] = []  # strong spot masks
    # assume these are the same for all panels in the detector (it need only be approximate for te purpose here)
    detdist = detector[0].get_distance()
    pixsize = detector[0].get_pixel_size()[0]
    fs_dim, ss_dim = detector[0].get_image_size()
    # FIXME: what about when same HKL indexed on different panels ?
    for refls, beam in zip(refls_at_colors, beams_of_colors):
        H, Hi, Q = refls_to_hkl(
            refls, detector, beam, crystal,  returnQ=True)

        color_data["panel"].append(list(refls['panel']))
        color_data["Pterms"].append(list(refls["intensity.sum.value"]))
        color_data["Q"].append(list(Q))
        color_data["H"].append(list(H))
        color_data["Hi"].append(list(map(tuple, Hi)))
        Qmag = np.linalg.norm(Q, axis=1)
        if twopi_conv:
            Qmag *= 2*np.pi

        x, y, _ = xyz_from_refl(refls)
        color_data["x"].append(x)
        color_data["y"].append(y)
        color_data["Qmag"].append(Qmag)

    ave_wave = np.mean([beam.get_wavelength() for beam in beams_of_colors])
    all_indexed_Hi = [tuple(h) for hlist in color_data["Hi"] for h in hlist]
    unique_indexed_Hi = set(all_indexed_Hi)

    all_x, all_y, all_H = [], [], []
    patches = {}
    bboxes = []
    panel_ids = []
    all_spot_Pterms = []  # NOTE this is for the preliminary merging code
    all_spot_Pterms_color_idx = []
    all_kept_Hi = []
    bbox_masks = []
    for H in unique_indexed_Hi:
        x_com = 0
        y_com = 0
        Qmag = 0
        n_counts = 0
        seen_pids = []  # how many different panel ids for this HKL - should be same pid across all colors
        in_multiple_panels = False
        spot_Pterms = []
        spot_Pterms_color_idx = []
        for i_color in range(len(beams_of_colors)):
            in_color = H in color_data["Hi"][i_color]
            if not in_color:
                continue
            idx = color_data["Hi"][i_color].index(H)

            pid = color_data["panel"][i_color][idx]
            if pid not in seen_pids:
                seen_pids.append(pid)
            if len(seen_pids) > 1:
                in_multiple_panels = True
                break
            x_com += color_data["x"][i_color][idx] - 0.5
            y_com += color_data["y"][i_color][idx] - 0.5
            Qmag += color_data["Qmag"][i_color][idx]
            n_counts += 1
            spot_Pterms.append(color_data["Pterms"][i_color][idx])
            spot_Pterms_color_idx.append(i_color)

        if in_multiple_panels:
            continue
        assert len(seen_pids) == 1  # sanity check, will remove
        PID = seen_pids[0]
        panel_ids.append(PID)
        Qmag = Qmag / n_counts
        all_x.append(x_com / n_counts)
        all_y.append(y_com / n_counts)
        all_spot_Pterms.append(spot_Pterms)
        all_spot_Pterms_color_idx.append(spot_Pterms_color_idx)
        all_kept_Hi.append(H)
        x_com = x_com / n_counts
        y_com = y_com / n_counts

        rad1 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag-delta_q*.5)*ave_wave/4/np.pi))
        rad2 = (detdist/pixsize) * np.tan(2*np.arcsin((Qmag+delta_q*.5)*ave_wave/4/np.pi))
        delrad = rad2-rad1

        i1 = int(max(x_com - delrad/2., 0))
        i2 = int(min(x_com + delrad/2., fs_dim))
        j1 = int(max(y_com - delrad/2., 0))
        j2 = int(min(y_com + delrad/2., ss_dim))

        mask_region = panel_masks[PID][j1:j2, i1:i2]
        bbox_masks.append(mask_region)
        bboxes.append((i1, i2, j1, j2))   # i is fast scan, j is slow scan

        if ret_patches:
            R = plt.Rectangle(xy=(x_com-delrad/2., y_com-delrad/2.),
                              width=delrad,
                              height=delrad,
                              **kwargs)
            try:
                patches[PID].append(R)
            except KeyError:
                patches[PID] = [R]

        if ret_Pvals:

            sub_data = data[PID, j1:j2, i1:i2]
            sub_mask = ((~allspotmask[PID]) * bad_pixel_mask[PID])[j1:j2, i1:i2]

            tilt, bgmask, coeff = integrate_utils.tilting_plane(
                sub_data,
                mask=sub_mask,
                zscore=2)

            data_to_be_integrated = sub_data - tilt

            int_mask = np.zeros(data_to_be_integrated.shape, bool)
            for i_color in spot_Pterms_color_idx:
                int_mask = np.logical_or(int_mask, color_integration_masks[i_color][PID][j1:j2, i1:i2])

            Yobs = (data_to_be_integrated*int_mask).sum() / gain
            integrated_Hi.append(Yobs)

    #unique_indexed_Hi = set(all_indexed_Hi)
    return_lst = [all_kept_Hi, bboxes, panel_ids, bbox_masks]

    assert len(all_kept_Hi) == len(bboxes) == len(panel_ids)

    if ret_patches:
        return_lst.append(patches)
    if ret_Pvals:
        return_lst += [all_spot_Pterms, all_spot_Pterms_color_idx, integrated_Hi]
    return return_lst


def strong_spot_mask(refl_tbl, detector):
    """

    :param refl_tbl:
    :param detector:
    :return:  strong spot mask same shape as detector (multi panel)
    """
    # dxtbx detector size is (fast scan x slow scan)
    nfast, nslow = detector[0].get_image_size()
    n_panels = len(detector)
    is_strong_pixel = np.zeros((n_panels, nslow, nfast), bool)
    panel_ids = refl_tbl["panel"]
    # mask the strong spots so they dont go into tilt plane fits
    for i_refl, (x1, x2, y1, y2, _, _) in enumerate(refl_tbl['bbox']):
        i_panel = panel_ids[i_refl]
        bb_ss = slice(y1, y2, 1)
        bb_fs = slice(x1, x2, 1)
        is_strong_pixel[i_panel, bb_ss, bb_fs] = True
    return is_strong_pixel


class FormatInMemory:
    """
    this class is a special image type
    necessary to create dxtbx imagesets and
    datablocks from numpy array images
    and masks.
    """
    def __init__(self, image, mask=None):
        self.image = image
        if image.dtype != np.float64:
            self.image = self.image.astype(np.float64)
        if mask is None:
            self.mask = np.ones_like(self.image).astype(np.bool)
        else:
            assert (mask.shape == image.shape)
            assert(mask.dtype == bool)
            self.mask = mask

    def get_raw_data(self):
        if len(self.image.shape)==2:
            return flex.double(self.image)
        else:
            return tuple([flex.double(panel) for panel in self.image])

    def get_mask(self, goniometer=None):
        if len(self.image.shape)==2:
            return flex.bool(self.mask)
        else:
            return tuple([flex.bool(panelmask) for panelmask in self.mask])




def explist_from_numpyarrays(image, detector, beam, mask=None):
    """
    So that one can do e.g.
    >> dblock = datablock_from_numpyarrays( image, detector, beam)
    >> refl = flex.reflection_table.from_observations(dblock, spot_finder_params)
    without having to utilize the harddisk
    :param image:  numpy array image, or list of numpy arrays
    :param mask:  numpy mask, should be same shape format as numpy array
    :param detector: dxtbx detector model
    :param beam: dxtbx beam model
    :return: datablock for the image
    """
    if isinstance( image, list):
        image = np.array( image)
    if mask is not None:
        if isinstance( mask, list):
            mask = np.array(mask).astype(bool)
    I = FormatInMemory(image=image, mask=mask)
    reader = MemReader([I])
    iset_Data = ImageSetData(reader, None) # , masker)
    iset = ImageSet(iset_Data)
    iset.set_beam(beam)
    iset.set_detector(detector)
    explist = ExperimentListFactory.from_imageset_and_crystal(iset, None)
    return explist


def refls_from_sims(panel_imgs, detector, beam, thresh=0, filter=None, panel_ids=None,
                    max_spot_size=1000, **kwargs):
    """
    This is for converting the centroids in the noiseless simtbx images
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
    from dials.algorithms.spot_finding.finder import pixel_list_to_reflection_table

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
        img_sz = detector[int(pid)].get_image_size()  # for some reason the int cast is necessary in Py3
        flex_img = flex.double(img)
        flex_img.reshape(flex.grid(img_sz))

        flex_mask = flex.bool(mask)
        flex_mask.resize(flex.grid(img_sz))
        pl = PixelList(0, flex.double(img), flex.bool(mask))
        plab.add(pl)

        pxlst_labs.append(plab)

    El = explist_from_numpyarrays(panel_imgs, detector, beam)
    iset = El.imagesets()[0]
    refls = pixel_list_to_reflection_table(
        iset, pxlst_labs,
        min_spot_size=1,
        max_spot_size=max_spot_size,  # TODO: change this ?
        filter_spots=FilterRunner(),  # must use a dummie filter runner!
        write_hot_pixel_mask=False)[0]

    return refls


def refls_from_sims_OLD(panel_imgs, detector, beam, thresh=0, filter=None, panel_ids=None, **kwargs):
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
        img_sz = detector[int(pid)].get_image_size()  # for some reason the int cast is necessary in Py3
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


def strong_spot_mask_dials(refl_tbl, img_size, as_composite=True):
    from dials.algorithms.shoebox import MaskCode
    Nrefl = len( refl_tbl)
    masks = [ refl_tbl[i]['shoebox'].mask.as_numpy_array()
              for i in range(Nrefl)]
    code = MaskCode.Foreground.real

    x1, x2, y1, y2, z1, z2 = zip(*[refl_tbl[i]['shoebox'].bbox
                                   for i in range(Nrefl)])
    if not as_composite:
        spot_masks = []
    spot_mask = np.zeros(img_size, bool)
    for i1, i2, j1, j2, M in zip(x1, x2, y1, y2, masks):
        slcX = slice(i1, i2, 1)
        slcY = slice(j1, j2, 1)
        spot_mask[slcY, slcX] = M & code == code
        if not as_composite:
            spot_masks.append(spot_mask.copy())
            spot_mask *= False
    if as_composite:
        return spot_mask
    else:
        return spot_masks

