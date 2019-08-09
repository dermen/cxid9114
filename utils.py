
import os
import cPickle
import numpy as np
import h5py
from dials.array_family import flex
from dxtbx.imageset import MemReader #, MemMasker
from dxtbx.datablock import DataBlockFactory
from dxtbx.imageset import ImageSet, ImageSetData
from dxtbx.model.experiment_list import ExperimentListFactory


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
    #masker = MemMasker([I])
    iset_Data = ImageSetData(reader, None) # , masker)
    iset = ImageSet(iset_Data)
    iset.set_beam(beam)
    iset.set_detector(detector)
    explist = ExperimentListFactory.from_imageset_and_crystal(iset, None)
    return explist


def datablock_from_numpyarrays(image, detector, beam, mask=None):
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
    #masker = MemMasker([I])
    iset_Data = ImageSetData(reader, None) #, masker)
    iset = ImageSet(iset_Data)
    iset.set_beam(beam)
    iset.set_detector(detector)
    dblock = DataBlockFactory.from_imageset([iset])[0]
    return dblock


def open_flex(filename):
    """unpickle the flex file which requires flex import"""
    with open(filename, "r") as f:
        data = cPickle.load(f)
    return data


def save_flex(data, filename):
    """save pickle"""
    with open(filename, "w") as f:
        cPickle.dump(data, f)



def smooth(x, beta=10.0, window_size=11):
    """
    https://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html
    
    Apply a Kaiser window smoothing convolution.

    Parameters
    ----------
    x : ndarray, float
        The array to smooth.

    Optional Parameters
    -------------------
    beta : float
        Parameter controlling the strength of the smoothing -- bigger beta
        results in a smoother function.
    window_size : int
        The size of the Kaiser window to apply, i.e. the number of neighboring
        points used in the smoothing.

    Returns
    -------
    smoothed : ndarray, float
        A smoothed version of `x`.
    """

    # make sure the window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # apply the smoothing function
    s = np.r_[x[window_size - 1:0:-1], x, x[-1:-window_size:-1]]
    w = np.kaiser(window_size, beta)
    y = np.convolve(w / w.sum(), s, mode='valid')

    # remove the extra array length convolve adds
    b = int((window_size - 1) / 2)
    smoothed = y[b:len(y) - b]

    return smoothed


def is_outlier(points, thresh=3.5):
    """
    http://stackoverflow.com/a/22357811/2077270

    Returns a boolean array with True if points are outliers and False
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def random_rotation(deflection=1.0, randnums=None):
    r"""
    Creates a random rotation matrix.

    TODO: documentation

    Arguments:
        deflection (float): the magnitude of the rotation. For 0, no rotation; for 1, competely random
                            rotation. Small deflection => small perturbation.
        randnums (numpy array): 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    Returns:
        (numpy array) Rotation matrix
    """
    # from
    # http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    vec = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    rot = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    mat = (np.outer(vec, vec) - np.eye(3)).dot(rot)
    return mat.reshape(3, 3)

