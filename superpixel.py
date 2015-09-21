from skimage.segmentation import slic
# from skimage import io
from statistics import mean, median, standard_deviation, inverse_normal_cdf, interquartile_range
import numpy as np


def superpixel_buckets(image, superpixel_labels):
    buckets = {}
    for r in xrange(0, len(image)):
        row = image[r]
        for c in xrange(0, len(row)):
            col = row[c]
            if superpixel_labels[r][c] not in buckets:
                buckets[superpixel_labels[r][c]] = []
            buckets[superpixel_labels[r][c]].append(col.tolist())
    return buckets


def medians(superpixel_buckets):
    medians = {}
    for i, pixel_list in superpixel_buckets.iteritems():
        medians[i] = [median([pixel[d] for pixel in pixel_list]) for d in (0, 1, 2)]
    return medians


def filter_out_item(d, item):
    i = item
    if type(item) == type(tuple()):
        i = list(item)
    filtered = {}
    for k, v in d.iteritems():
        if not v == i:
            filtered[k] = v
    return filtered


def mask_from_superpixel_labels(superpixel_labels, labels_to_keep_dict):
    mask = np.zeros(superpixel_labels.shape, np.uint8)
    for r in xrange(0, len(superpixel_labels)):
        row = superpixel_labels[r]
        for c in xrange(0, len(row)):
            if row[c] in labels_to_keep_dict:
                mask[r][c] = 255
    return mask


def in_range(src, lower, upper):
    return lower[0] <= src[0] <= upper[0] and lower[1] <= src[1] <= upper[1] and lower[2] <= src[2] <= upper[2]


def filter_in_range(d, lower, upper, keep=True):
    filtered = {}
    for k, v in d.iteritems():
        if in_range(v, lower, upper) == keep:
            filtered[k] = v
    return filtered


def superpixels(image):
    # image = io.imread(url)
    numSegments = 400
    segments = slic(image, n_segments=numSegments, slic_zero=True)
    # io.imsave("segmented{0}_{1}.png".format(i, numSegments), marked)
    return segments


