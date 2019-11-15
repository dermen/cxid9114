
import numpy as np
from cxid9114 import utils


def tilting_plane(img, mask=None, zscore=2):
    """
    fit tilting plane to img data, used for background subtraction of spots
    :param img:  numpy image
    :param mask:  boolean mask, same shape as img, True is good pixels
        mask should include strong spot pixels and bad pixels, e.g. zingers
    :param zscore: modified z-score for outlier detection, lower increases number of outliers
    :return: tilting plane, same shape as img
    """
    Y,X = np.indices(img.shape)
    YY,XX = Y.ravel(), X.ravel()

    img1d = img.ravel()

    if mask is None:
        mask = np.ones( img.shape, bool)
    mask1d = mask.ravel()

    out1d = np.zeros( mask1d.shape, bool)
    out1d[mask1d] = utils.is_outlier( img1d[mask1d].ravel(), zscore)
    out2d = out1d.reshape (img.shape)

    fit_sel = np.logical_and(~out2d, mask)  # fit plane to these points, no outliers, no masked
    x,y,z = X[fit_sel], Y[fit_sel], img[fit_sel]
    guess = np.array([np.ones_like(x), x, y ] ).T
    coeff, r, rank, s = np.linalg.lstsq(guess, z)
    ev = (coeff[0] + coeff[1]*XX + coeff[2]*YY )
    return ev.reshape(img.shape), out2d, coeff


class Integrator:

    def __init__(self, data, gain=1, int_radius=8):
        self.gain = gain
        self.data = data/gain
        self.int_radius = int_radius

    def integrate_bbox_dirty(self, bbox):

        i1, i2, j1, j2 = bbox

        sub_data = self.data[j1:j2, i1:i2]

        tilt, bgmask, coeff = tilting_plane(
            sub_data,
            zscore=2)
        # NOTE: bgmask shows the pixels that were masked before fitting the tilt plane..
        bgmask = np.logical_not(bgmask)

        data_to_be_integrated = sub_data - tilt

        cent = (j1+j2)*.5, (i1+i2)*.5

        X, Y = np.meshgrid(range(i1, i2), range(j1, j2))

        R = np.sqrt((Y - cent[0]) ** 2
                  + (X - cent[1]) ** 2)

        rad = self.int_radius
        if self.int_radius*2 > j2-j1 or self.int_radius*2 >= i2-i1:
            rad = min(j2-j1, i2-i1) - 1
        int_mask = R < rad

        # from Leslie '99
        m = int_mask.sum()
        n = bgmask.sum()
        Is = (data_to_be_integrated*int_mask).sum()
        Ibg = m/n * sub_data[bgmask].sum()
        noise = (Is + Ibg + m/n * Ibg)

        #from IPython import embed
        #embed()
    
        return Is, Ibg, noise


