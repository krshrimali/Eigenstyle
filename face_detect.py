import cv2
import urllib
import numpy as np
import product_catalog
from statistics import mean, median, standard_deviation, inverse_normal_cdf, interquartile_range
from math import sqrt
import superpixel
from skimage.segmentation import mark_boundaries


DEFAULT_FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_FACE_CASCADE = cv2.CascadeClassifier(DEFAULT_FACE_CASCADE_PATH)


def cv_open_image_from_url(imagePath):
    # Read the image
    req = urllib.urlopen(imagePath)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)  # 'load it as it is'
    return image


MEDIAN_FACE = (118, 18, 36, 36)


def distance(a, b):
    return sqrt(sum([(a[i]-b[i])*(a[i]-b[i]) for i in xrange(len(a))]))


def best_face(faces):
    # distances from median face
    distances = [distance(face, MEDIAN_FACE) for face in faces]
    return faces[np.argmin(distances)]


def detect_faces(image, face_cascade=DEFAULT_FACE_CASCADE, scale_factor=1.1, min_neighbors=1, min_size=(25, 25), max_size=(50, 50), flags=cv2.cv.CV_HAAR_SCALE_IMAGE):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        maxSize=max_size,
        flags=flags
    )
    # convert numpy array to list
    if len(faces) > 0:
        faces = faces.tolist()
    else:
        faces = []
    return faces


def detect_face(image, face_cascade=DEFAULT_FACE_CASCADE):
    faces = detect_faces(image, face_cascade)
    # remove unreasonable faces
    faces = [face for face in faces if face[1] < 40]  # y value reasonable?
    # pick best face if there is more than 1
    if len(faces) > 1:
        face = best_face(faces)
    elif len(faces) == 1:
        face = faces[0]
    else:
        face = None
    return face


def detect_all_faces(url_list, display_mode=False):
    all_faces = []
    face_urls = []
    noface_urls = []
    for i in xrange(0, len(url_list)):
        image = cv_open_image_from_url(url_list[i])
        face = detect_face(image)
        if face is not None:
            print "{:0.2f} Found 1 face!".format(i / float(len(url_list)))
            all_faces.append(face)
            if display_mode:
                draw_box(image, face, (0, 255, 0))
                draw_box(image, dress_box(face), (255, 0, 0))
            face_urls.append(url_list[i])
        else:
            print "{:0.2f} Found 0 face!".format(i / float(len(url_list)))
            noface_urls.append(url_list[i])
        if display_mode:
            cv2.imshow(str(i), image)
            cv2.waitKey(0)
    print "all_faces", all_faces
    print "face_urls", face_urls
    print "noface_urls", noface_urls
    return all_faces, face_urls, noface_urls


IMAGE_WIDTH = 270
IMAGE_HEIGHT = 405
IMAGE_ASPECT_RATIO = 270/float(405)


def dress_box(face, im_dimensions=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    im_height, im_width = im_dimensions
    im_aspect = im_width / float(im_height)
    (x, y, w, h) = face
    bottom_of_face = y+h
    y_ = bottom_of_face  # bottom of face
    h_ = im_height-bottom_of_face  # bottom of face to bottom of image
    w_ = int(im_aspect * h_)  # compute width so that dress box is same aspect ratio as image
    x_ = (im_width/2)-(w_/2)  # centered in x
    return x_, y_, w_, h_


def dress_boxes(faces):
    return [dress_box(face) for face in faces]


def draw_box(image, box, color=(0, 255, 0)):
    (x, y, w, h) = box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def draw_boxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        draw_box(image, box, color)


def print_box_stats(boxes):
    xs = [f[0] for f in boxes]
    ys = [f[1] for f in boxes]
    widths = [f[2] for f in boxes]
    heights = [f[3] for f in boxes]
    print
    print "min x", min(xs)
    print "median x", median(xs)
    print "max x", max(xs)
    print
    print "min y", min(ys)
    print "median y", median(ys)
    print "max y", max(ys)
    print
    print "min w", min(widths)
    print "median w", median(widths)
    print "max w", max(widths)
    print
    print "min h", min(heights)
    print "median h", median(heights)
    print "max h", max(heights)


def skin_detect(image):
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([10, 48, 80], dtype = "uint8")
    # lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")

    # convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the specified upper and lower boundaries
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    noskinMask = cv2.bitwise_not(skinMask)  # TODO: ****** not quite right ********

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(image, image, mask = skinMask)
    noskin = cv2.bitwise_and(image, image, mask = noskinMask)

    # show the skin in the image along with the mask
    cv2.imshow("images", np.hstack([image, skin, noskin]))
    cv2.waitKey(0)


def skin_detect2(image, marks=False):
    lower = [10, 48, 80]
    upper = [20, 255, 255]
    superpixel_labels = superpixel.superpixels(image)
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    buckets = superpixel.superpixel_buckets(hsvim, superpixel_labels)
    skinpatches = superpixel.medians(buckets)
    # skinpatches = superpixel.filter_out_item(skinpatches, [0, 0, 255])  # remove bulk of white background superpixels
    skinpatches = superpixel.filter_in_range(skinpatches, lower, upper, keep=False)
    skinmask = superpixel.mask_from_superpixel_labels(superpixel_labels, skinpatches)
    skin = cv2.bitwise_and(image, image, mask=skinmask)
    if marks:
        skin = mark_boundaries(skin, superpixel_labels)

    cv2.imshow("images", np.hstack([image, skin]))
    cv2.waitKey(0)
    return skin


def skin(marks=False):
    for u in product_catalog.FACE_URLS:
        skin_detect2(cv_open_image_from_url(u), marks=marks)


# skin()

# skin(marks=True)

# detect_all_faces(product_catalog.FACE_URLS, display_mode=True)
