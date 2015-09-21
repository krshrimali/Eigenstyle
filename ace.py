from math import sqrt
import random
import numpy as np


def calc_ace(img, samples=500, slope=10, limit=1000):
    # init rand seed
    random.seed()
    # get image height and width
    height, width, channels = img.shape
    # get random sample of pixel coordinates from image
    sampled_coordinates = create_random_pair(samples, width, height)
    rscore = np.zeros_like(img, np.float64)
    # chromatic/spacial adjustment
    for row in xrange(height):
        for col in xrange(width):
            # calculate r score
            rscore_sum = [0.0, 0.0, 0.0]
            denominator = 0.0
            for x, y in sampled_coordinates:
                dist = calc_euclidean(col, row, x, y)
                if dist < height/5.0:
                    continue
                for channel in xrange(channels):
                    print "rscore_sum[channel] {0} img[row][col][channel] {1} img[y][x][channel] {2} dist {3}".format(rscore_sum[channel], img[row][col][channel], img[y][x][channel], dist)
                    rscore_sum[channel] += calc_saturation(img[row][col][channel] - img[y][x][channel], slope, limit) / dist
                denominator += limit/dist
            rscore_sum = [score/denominator for score in rscore_sum]
            rscore[row][col] = np.array(rscore_sum, np.float64)
    maxs = [rscore[:, :, channel].max() for channel in xrange(channels)]
    mins = [rscore[:, :, channel].min() for channel in xrange(channels)]
    # dynamic tone reproduction scaling
    for row in xrange(height):
        for col in xrange(width):
            # scaling
            for channel in xrange(channels):
                img[row][col][channel] = linear_scaling(rscore[row][col][channel], maxs[channel], mins[channel])


def calc_euclidean(ax, ay, bx, by):
    return sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by))


def calc_saturation(diff, slope, limit):
    ret = diff * slope
    if ret > limit:
        return limit
    if ret < -limit:
        return -limit
    return ret


def linear_scaling(r, max, min):
    slope = 255.0/(max-min)
    return int((r-min)*slope)


def linear_scaling2(r, max, min):
    slope = max/127.5
    return int(r*slope+127.5)


def create_random_pair(size, x, y):
    return [(random.randrange(x), random.randrange(y)) for i in xrange(size)]

# import product_catalog
# import face_detect
# import ace
# import os
# import os.path
# import cv2
# u = product_catalog.FACE_URLS[42]
# custim = [os.path.join('cust_im', c) for c in os.listdir('cust_im') if os.path.splitext(c)[1] == '.jpg']
# custim
# im = cv2.imread(custim[0])
# im
# ace.calc_ace(im, samples=500, slope=10, limit=1000)
