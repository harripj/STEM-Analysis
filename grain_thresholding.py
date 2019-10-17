#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:01:51 2018

@author: pjh523
"""

from scipy import optimize as _optimize, ndimage as _ndi, signal as _signal
from skimage import segmentation as _segmentation, measure as _measure, \
                    exposure as _exposure, filters as _filters, \
                    feature as _feature, morphology as _morphology
from matplotlib import pyplot as _plt
import numpy as _np
from scipy import ndimage as _ndi

import logging as _logging
from types import FunctionType as _FunctionType
from collections import Iterable as _Iterable

try:
    from peak_finder import create_guess_from_peak_positions as _cgfpp
    from models import Gaussian as _Gauss
except ImportError:
    _logging.warning('Some modules have not been loaded.' \
                    + ' Some functions may be unavailable.')

def radial_threshold(image, coords, radii, npts=128):
    '''

    UNFINISED
    Perform a radial line profile around each point in coords to establish a threshold.

    Parameters
    ----------
    image: numpy.ndarray
        Image to threshold.
    coords: numpy.ndarray, shape Nximage.ndim
        Coordinates of features to threshold.
    Radii: array-like
        Radius values to consider.
    '''
    # format inputs
    radii = _np.asarray(radii)

    # full circle
    theta = _np.linspace(0, 2*_np.pi, npts)
    
    # for each coordinate
    for coord in coords:
        # do multiple line profiles along radius
        lp = _np.array([r*_np.array((_np.sin(theta), _np.cos(theta))) + coord[:, _np.newaxis] for r in radii])
        profiles = _np.split(_ndi.map_coordinates(image, _np.concatenate(lp, axis=1)), radii.size)
# TODO- work out some kind of threshold with these profiles...

def threshold_otsu_power(image, power=1./10):
    '''

    Compute Otsu's threshold on power scaled data.

    Parameters
    ---------
    image: numpy.ndarray
        Image to threshold.
    power: float, default is 1./10
        Power scaling factor.

    Returns
    -------
    threshold: float

    '''
    # make data copy to ensure array safety
    image = _np.array(image, dtype=float)
    offset = image.min()
    # shift data range to avoid negative numbers
    image -= offset
    # compute threshold
    threshold = _filters.threshold_otsu(_np.power(image, power))
    # return threshold in original scale
    return _np.power(threshold, 1./power) + offset

def binary_from_canny(image, radius=10, **kwargs):
    '''
    Produces a filled binary image from a Canny edge detector which can then be lablled and analysed
    etc.

    Parameters
    ----------
    image: np.ndarray, ndim=2
        Greyscale image to apply Canny edge detector to.
    radius: int
        Disk radius to use for binary_closing operation.
    **kwargs:
        Arguments to pass to skimage.feature.canny function.

    Returns
    -------
    filled_binary: np.ndarray
        Binary image, same shape as input with edges detected by Canny filter, and subsequently filled.

    '''
    # normalise image to range [0,1], as type float
    image = _exposure.rescale_intensity(image.astype(float), out_range=(0.,1.))
    # detect edges, and close up any close neighbours
    edges = _feature.canny(image, **kwargs)
    edges = _morphology.binary_closing(edges, selem=_morphology.disk(1))
    # measure and label found edges
    rp = _measure.regionprops(_measure.label(edges))
    # initialise array to return
    out = _np.zeros_like(image, dtype=bool)
    # for each edge...
    for _rp in rp:
        # get image box
        _image = _rp.filled_image
        r0, c0, r1, c1 = _rp.bbox
        # pad by radius otherwise closing operation from edges will not be correct
        padded = _np.pad(_image, ((radius, radius),)*image.ndim, \
                         mode='constant', constant_values=False)
        # fill the edges
        closed = _morphology.binary_closing(padded, selem=_morphology.disk(radius))
        closed = closed[radius:-radius, radius:-radius]
        # add to output
        out[r0:r1, c0:c1] = closed
    return out

def threshold_otsu_median(image, nbins=256):
    '''
    Calculates a greyscale image threshold using Otsu's method but using the class median (histogram mode) as the intraclass descriptor.
    This algorithm works especially well for the edges of the image bright features.

    This '#1' version uses the class modes of the image histogram as the threshold as opposed to the class medians of the image in the '#2' version.
    As a result it is over 10x faster in testing.
    The thresholds obtained from '#1' and '#2' are extremely similar.

    Parameters
    ----------
    image: np.ndarray
        The image to threshold.
    nbins: int
        Number of bins for the histogram. Default is 256.

    Returns
    -------
    threshold: float
        The calculated threshold.
    '''
    # image needed as float, otherwise median calculations stall
    image = image.astype(float)
    hist, bin_centers = _exposure.histogram(image, nbins=nbins)

    # make sure dtype is float
    hist = hist.astype(float)
    bin_centers = bin_centers.astype(float)

    # class probabilities for all possible thresholds
    # NB. float-type conversion needed for python2 support, as hist.dtype == int
    weight1 = _np.cumsum(hist) / _np.sum(hist)
    weight2 = _np.cumsum(hist[::-1])[::-1] / _np.sum(hist) # cumulative sum backwards
    
    # class medians for all possible thresholds
    mode1 = bin_centers[_np.array([_np.argmax(hist[:i]) for i in range(1, hist.size)])]
    mode2 = bin_centers[_np.array([i + _np.argmax(hist[i:]) for i in range(1, hist.size)])]

    # median for whole image
    modeT = bin_centers[_np.argmax(hist)]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1]*(mode1-modeT)**2 + weight2[1:]*(mode2-modeT)**2

    idx = _np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def _otsu_median_variance(image, nbins=256):
    '''

    Performs main bulk of calculations for Otsu's median adapted threshold.

    Parameters
    ----------
    image: numpy.ndarray
        Greyscale image.
    nbins: int, default is 256
        Number of bins in histogram.

    Returns
    -------
    variance: numpy.ndarray, length=nbins-1
        Calculated intra-class variance
    bin_centers: numpy.ndarray, length=nbins
        Histogram bin locations.
    
    '''
    # image needed as float, otherwise median calculations stall
    image = _np.asarray(image, dtype=float)
    hist, bin_centers = _exposure.histogram(image, nbins=nbins)

    # class probabilities for all possible thresholds
    cumulative = _np.cumsum(hist)

    # sort all intensity values
    vals = _np.sort(image.ravel())

    # make hist float otherwise calculations will fail
    hist = hist.astype(float)
    weight1 = cumulative / _np.sum(hist)
    weight2 = _np.cumsum(hist[::-1])[::-1] / _np.sum(hist) # cumulative sum backwards
    
    # class medians for all possible thresholds
    median1 = _np.array([vals[:cumulative[i]][cumulative[i]//2] for i in range(0, hist.size-1)])
    median2 = _np.array([vals[cumulative[i-1]:][(vals.size-cumulative[i-1])//2] for i in range(1, hist.size)])
    
    # median for whole image
    medianT = vals[vals.size//2]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1]*(median1-medianT)**2 + weight2[1:]*(median2-medianT)**2

    return variance12, bin_centers


def threshold_otsu_median2(image, nbins=256):
    '''Calculates a greyscale image threshold using Otsu's method but using the class median (histogram mode) as the intraclass descriptor.
    This algorithm works especially well for the edges of the image bright features.

    This '#2' version actaully uses the class medians of the image as the threshold as opposed to the class modes of the histogram in the '#1' version.
    As a result it is over 10x slower in testing.
    The thresholds obtained from '#1' and '#2' are extremely similar.

    Parameters
    ----------
    image: np.ndarray
        The image to threshold.
    nbins: int
        Number of bins for the histogram. Default is 256.

    Returns
    -------
    threshold: float
        The calculated threshold.
    '''
    # calculate intra-class variance
    variance, bin_centers = _otsu_median_variance(image, nbins=nbins)
    # get maximum variance
    idx = _np.argmax(variance)
    # return threshold
    return bin_centers[:-1][idx]

def threshold_low_signal(image, nbins=256, max_iter=10000):
    '''

    Thresholds an image with low signal, ie. where one feature dominates the intensity histogram.
    Uses a combined approach of Otsu's median threshold and then minimum thresholding to find the threshold.

    Parameters
    ----------
    image: numpy.ndarray
        Image to threshold.
    nbins: int, default is 256
        Number of histogram bins.
    max_iter: int, default is 10000
        Maximum number of smoothing steps to perform.

    Returns
    -------
    threshold: float
        Calculated threshold value.
    
    '''
    # calculate intra-class variance
    variance, bin_centers = _otsu_median_variance(image, nbins=nbins)

    # basically perfroming skimage.filters.threshold_minimum algorithm
    # on variance histogram
    smooth_hist = variance.astype(_np.float, copy=False)

    # minimum algorithm
    for counter in range(max_iter):
        smooth_hist = _ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = _np.squeeze(_signal.argrelextrema(smooth_hist,
                                                         _np.greater_equal))
        # looking for two maxima
        if len(maximum_idxs) < 3:
            break

    if len(maximum_idxs) != 2:
        raise RuntimeError('Unable to find two maxima in histogram')
    elif counter == max_iter - 1:
        raise RuntimeError('Maximum iteration reached for histogram'
                           'smoothing')

    # Find lowest point between the maxima
    threshold_idx = _np.argmin(smooth_hist[maximum_idxs[0]: maximum_idxs[1]+1])
    # return threshold
    return bin_centers[:-1][maximum_idxs[0] + threshold_idx]

def revolve_arc(arr, radius=20., axis=(0,1)):
    '''
    Revolves an arc over each axis specified.
    Useful for background subtraction.

    Parameters
    ----------
    arr: numpy.ndarray
        Input array over which to revolve arc.
    radius: float. Default = 20.
        Arc radius.
    axis: int, tuple
        Axes over which to revolve arc. Can be singular or multiple.
        Default = (0,1).

    Returns
    -------
    out: numpy.ndarray
        Convolved array.

    '''
    # produce and normalise kernel
    a = _np.arange(-radius//2, +radius//2+1, dtype=float)
    kern1d = _np.cos(a/radius)
    kern1d /= kern1d.sum()

    # create output array
    out = _np.copy(arr).astype(float)

    if not isinstance(axis, _Iterable):
        # make iterable
        axis = (axis,)
    
    # for every axis specified
    for ax in axis:
        _ndi.convolve1d(_np.asarray(out), kern1d, output=out, axis=ax)

    return out

def calculate_full_grain_field(image, threshold=None, min_size=0, foreground=True):
    '''
    Calculates binary image according to threshold, removing grains smaller than min_size.

    Parameters
    ---------
    image: np.ndarray
        Image to threshold.
    threshold: int, float, numpy.ndarray, or func.
        The threshold. If func, then threshold(image) must return a number.
        If numpy array, should be dtype bool, in which case is treated as previously created mask and threshold value is calculated from True image values. Default is None, in which case threshold_otsu_median is used.
    min_size: int
        Remove grains smaller than this. Default is 5 pixels.
    foreground: bool
        Threshold features in foreground or background. Default is True (grains in foreground).

    Returns
    -------
    grain_field: np.ndarray
        Binary image with filtered, thresholded grains.

    '''
    # work out threshold, get corresponding binary image, and label the grains
    assert isinstance(threshold, \
                      (int, float, _FunctionType, _np.ndarray, type(None))), \
                      'threshold argument must be of type: ' \
                      + 'int, float, numpy.ndarray, func, or None.'
    
    # handle case where threshold is function
    if type(threshold) is _FunctionType:
        threshold =  threshold(image)
    elif isinstance(threshold, _np.ndarray):
        # convert to mask, should be type mask anyway...
        # numerical threshold is smallest value in mask
        assert threshold.shape == image.shape, \
            'If threshold is numpy array must be same shape as image.'
        if foreground:
            threshold = _np.min(image[threshold.astype(bool)])
        else:
            # if bg is ROI, then the boundary is the maximum value of bg
            threshold = _np.max(image[threshold.astype(bool)])
    elif threshold is None:
        # default case
        threshold = threshold_otsu_median(image)
    
    # ROI in foreground (intense), or background
    if foreground:
        grains = image > threshold
    else:
        grains = image < threshold
    
    return _morphology.remove_small_objects(grains, min_size=min_size)

def grains_by_hysteresis_threshold(image, nsigma=5, plot=False):
    '''
    Creates a grain binary mask from a greyscale image using the hysteresis 
    threshold.
    
    The algorithm is thus:
        - Compute histogram of image
        - Measure background intensity and fit Gaussian
        - Upper threshold of skimage.filters.apply_hysteresis_threshold is the 
            Otsu threshold of the image
        - Lower threshold is the maximum value of the BG gaussian fit + n*sigma
            that is still lower that Otsu's threshold
            
    This helps measure grain edges (which can be smaller than Otsu's threshold).
    
    Parameters
    ----------
    image: numpy.ndarray
        Greyscale image.
    nsigma: int
        Maximum value of BG sigma to use as lower threshold.
    plot: bool
        Whether to plot the histogram and fit.
        
    Returns
    -------
    mask: numpy.ndarray
        Binary mask of found grains.
        
    '''
    # make image of type float
    image = image.astype(float)
    # creat histogram of intensities
    vals, bins = _exposure.histogram(image)
    # create guess of background -> maximum of histogram corresponds to bg pixels
    guess = _cgfpp(vals, bins, [bins[vals.argmax()]])
    fp, _ = _optimize.curve_fit(_Gauss.vector, bins, vals, p0=guess)
    x0, A, sigma = fp
    
    # calcultae classical Otsu threshold of image
    thold_otsu = _filters.threshold_otsu(image)
    # the lower bound for hystereiss threshold is where the maximum of BG
    # Guassian fit x0+n*sigma that is smaller that Otsu threshold
    _nsigma = _np.arange(1, nsigma+1)*sigma+x0 
    _lower = _nsigma[(thold_otsu-_nsigma)>0].max()
    
    # compute grainmask
    mask = _filters.apply_hysteresis_threshold(image, _lower, thold_otsu)
    
    if plot:
        _plt.plot(bins, vals, label='Data')
        _plt.plot(bins, _Gauss.vector(bins, *fp), label='Fit')
        _plt.axvline(_lower, label='Lower threshold')
        _plt.axvline(thold_otsu, label='Upper (Otsu) threshold')
        _plt.legend()
        
    return mask

def binary_segmentation(binary, min_distance=5, markers=None, mask_only=True, use_distance=False):
    '''
    Perform watershed segmentation algorithm, as is described here [1].
    
    Parameters
    ----------
    binary: numpy.ndarray
        Binary image to watershed, fetures of interest are bright.
    min_distance: int
        Minimum distance allowed between found local peaks in distance transform.
        (skimage.feature.peak_local_max argument)
    markers: np.ndarray of int
        Labelled array, same shape as binary, which is to be used as the watershed
        drop points. Default is None, in which case markers are found from the
        Euclidean Distance Transform of the input binary mask.
    mask_only: bool
        Whether to use only watershed points which are within the mask. Default is True.
    use_distance: bool
        If True the watershed is performed on the distance transform.
        Otherwise watershed is performed on the (flat) binary input. Default is False.
        
    Returns
    -------
    labels: numpy.ndarray
        Labelled segmented image of input.
        
    [1] http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
    '''
    # do distance transform of mask
    distance = _ndi.distance_transform_edt(binary)
    
    if markers is not None:
        assert markers.shape == binary.shape, 'Markers must be same shape as binary.'
    else:
        # find peaks in distance transform, and return as image array
        local_maxi = _feature.peak_local_max(distance, indices=False, \
                                            min_distance=min_distance, labels=binary)
        # label found local maximum
        markers = _measure.label(local_maxi)

    if mask_only:
            # remove markers outside of mask...
            # ...this avoids single pixels filling up
            markers[_np.logical_not(binary)] = 0
    
    # do watershed transform on about the edt (now as valleys), with the found local
    # maximum as the initial filling points, and label only the original binary mask
    if use_distance:
        # do watershed on edt
        labels = _morphology.watershed(_np.negative(distance), markers, mask=binary)
    else:
        # do watershed on flat mask (no gradient descent)
        labels = _morphology.watershed(_np.logical_not(binary), markers, mask=binary)

    return labels

def analyse_all_grain_masks(binary_images, intensity_images=None, min_size=6,
                            pixel_sizes=None):
    '''
    Measure the skimage.measure.regionproperties of grains in each binary
    grain mask. 
    
    Parameters
    ----------
    binary_images: list-like
        List of grain masks.
    intensity_images: list-like
        List of associated intensity images.
    min_size: int
        Minimum pixel area to count as grain. Default is 6.
        Wrapper to measure_and_tidy_up function.
    pixel_sizes: array-like
        List of the real-area values of one pixel in the image.
        This information is defined as a class attribute of each RegionProps.
        If defined, this argumnet must have same length as binary_images.
        Default is None.
        eg. for DM3 data:
            pixel_sizes = [(dm3.pxsize[0] * dm3_functions.DM3_SCALE_DICT[dm3.pxsize[1]]) \
                           for dm3 in dm3_images]
        
    Returns
    -------
    rp: list of skimage.measure.RegionProps
    
    '''
    if intensity_images is not None:
        assert len(intensity_images) == len(binary_images), \
            'Each binary_image must have and associated intensity_image.'
    if pixel_sizes is not None:
        assert len(pixel_sizes) == len(binary_images), \
            'Each image must have an associated pixel_size if arg. is defined.'
    # holder for regionproperties
    rp = []
    for _index, image in enumerate(binary_images):
        assert type(image) is _np.ndarray, 'Not an image: {}'.format(image)
        
        if intensity_images is not None:
            _ii = binary_images[_index]
        else:
            _ii = None
        
        # add pixel_size attribute to grain if defined, otherwise None.
        if pixel_sizes is None:
            labelled, _rp = measure_and_tidy_up(image, min_size=min_size, \
                                                intensity_image=_ii, \
                                                pixel_size=None)
        else:
            labelled, _rp = measure_and_tidy_up(image, min_size=min_size, \
                                                intensity_image=_ii, \
                                                pixel_size=pixel_sizes[_index])
        # append to list
        rp.extend(_rp)
    
    return rp

def plot_grain_mask(image, grain_mask, cmap='gray', rgb_idx=0, alpha=0.5):
    '''
    Produce a quick figure showing the found grains on top of the original image.
    
    Parameters
    ----------
    image: numpy.ndarray, ndim=2
        Original data image.
    grain_mask: numpy.ndarray, ndim=2, binary
        Mask where grains are 1, background is 0.
    cmap: str
        matplotlib colormap, default is 'gray'.
    rgb_idx: int
        Plot the grains as red=0, green=1, blue=2.
    alpha: float
        Alpha value for grain_mask, must be in range 0:1.
        
    Returns
    -------
    
    f, ax: tuple of matplotlib figure and plotted axes.
    
    '''
    assert image.shape == grain_mask.shape, \
        'image and grain_mask are not the same shape.'
    assert rgb_idx >=0 and rgb_idx <= 2, 'rgb_idx must be positive and <=2.'
    
    f, ax = _plt.subplots()
    # image plot
    ax.matshow(image, cmap=cmap)
    
    sx, sy = image.shape
    overlay = _np.zeros((sx, sy, 4))
    overlay[grain_mask, rgb_idx] = 1
    overlay[grain_mask, 3] = alpha
    # grain mask overlay plot
    ax.imshow(overlay)
    
    return f, ax
