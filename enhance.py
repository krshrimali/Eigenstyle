import cv2
from math import floor, ceil
import os
import face_detect
from colorcorrect.algorithm import automatic_color_equalization
import subprocess
import re


BRISQUE = 'brisque_revised/brisquequality'
BRISQUE_SCORE_PATTERN = re.compile('score in main file is given by:(-?\\d+\\.\\d+)')


def brisque_quality_score(image_path):
    process = subprocess.Popen([BRISQUE, '-im', image_path], stdout=subprocess.PIPE)
    process_output = process.communicate()[0]
    match = BRISQUE_SCORE_PATTERN.search(process_output)
    if not len(match.groups()) == 1:
        print "{}: BRISQUE output was: {}".format(image_path, process_output)
        return None
    # otherwise we're good so return score
    return float(match.groups()[0])


def simplest_color_balance(src, percent=1):
    # assert(input.channels() ==3)
    # assert(percent > 0 && percent < 100)

    half_percent = float(percent) / 200.0

    channels = cv2.split(src)
    out = []
    for channel in channels:
        # find the low and high percentile values (based on the input percentile)
        flat = channel.ravel().tolist()
        flat.sort()
        lowval = flat[int(floor(float(len(flat)) * half_percent))]
        highval = flat[int(ceil(float(len(flat)) * (1.0-half_percent)))]

        # saturate below the low percentile and above the high percentile
        # channel = cv2.threshold(channel, highval, -1, cv2.THRESH_TRUNC) # truncate values to max of highval
        # for row in channel:
        #     for c in xrange(len(row)):
        #         if row[c] < lowval
        channel = cv2.max(channel, lowval)
        channel = cv2.min(channel, highval)

        # scale the channel
        channel = cv2.normalize(channel, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        out.append(channel)

    out = cv2.merge(out)
    return out


def auto_crop_hsv(src):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    threshval, thresh = cv2.threshold(v, 20, 255, cv2.THRESH_BINARY)  # simple global thresh to get anything "blackish"
    projx = [sum(col) for col in thresh.transpose()]
    projy = [sum(row) for row in thresh]
    nonzero = lambda value: value > 0
    x1 = first_index_of(projx, nonzero)
    x2 = last_index_of(projx, nonzero)
    y1 = first_index_of(projy, nonzero)
    y2 = last_index_of(projy, nonzero)
    cropped = src[y1:y2+1, x1:x2+1]
    return cropped


def first_index_of(l, pred):
    for i, v in enumerate(l):
        if pred(v):
            return i
    return None


def last_index_of(l, pred):
    for i, v in reversed(list(enumerate(l))):
        if pred(v):
            return i
    return None


def dress_box2(face, im_dimensions):
    im_height, im_width = im_dimensions
    (x, y, w, h) = face
    bottom_of_face = y+h
    y_ = bottom_of_face  # bottom of face
    h_ = im_height-bottom_of_face  # bottom of face to bottom of image
    w_ = int(w*4.5)  # based on face width
    x_ = x+(w/2)-(w_/2)  # centered below face
    return x_, y_, w_, h_


def crop_to_human(im, faces, dresses):
    # preserve ~1 face height above top face and crop off the rest
    top_face_y = min([f[1] for f in faces])
    max_face_height = max([f[3] for f in faces])
    y1 = max(0, top_face_y - max_face_height)
    # preserve roughly the area horizontally covered by the people's bodies (approximate width below each face)
    dress_bounds = bounding_box(dresses)
    x1 = dress_bounds[0]
    x2 = dress_bounds[2] + x1
    x1 = max(0, x1)
    x2 = min(im.shape[1] - 1, x2)
    im = im[y1:, x1:x2 + 1]
    return im


def auto_fix(im):
    im = auto_crop_hsv(im)
    min_face_size = (int(im.shape[0]*0.05), int(im.shape[1]*0.05))
    faces = face_detect.detect_faces(im, min_neighbors=5, min_size=min_face_size, max_size=None)
    # keep faces above the fold
    faces = [face for face in faces if face[1] < im.shape[0]*0.4]
    # find dresses
    dresses = [dress_box2(face, im.shape[:2]) for face in faces]
    if len(faces) > 0:
        im = crop_to_human(im, faces, dresses)
    im = simplest_color_balance(im)
    #im = automatic_color_equalization(im)
    # face_detect.draw_boxes(im, faces)
    # face_detect.draw_boxes(im, dresses, (255, 0, 0))
    if len(faces) == 0:
        face_score = 1.0/2.5
    else:
        face_score = 1.0/len(faces)
    print "{0}".format(face_score)
    return im


def bounding_box(boxes):
    xy = [min([b[i] for b in boxes]) for i in (0, 1)]
    wh = [max([b[i]+b[i-2] for b in boxes])-xy[i-2] for i in (2, 3)]
    return tuple(xy+wh)


def fix_dir(dir):
    files = os.listdir(dir)
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            print file
            im = cv2.imread(os.path.join(dir, file))
            im = auto_fix(im)
            outfile = os.path.splitext(file)[0]+"_fixed.png"
            cv2.imwrite(os.path.join(dir, outfile), im)


# import enhance
# import os
# import os.path
# import cv2
# custim = [i for i in os.listdir('cust_im') if os.path.splitext(i)[1] == '.jpg']
# im = cv2.imread(custim[1])
# im = enhance.auto_fix(im)
# faces = DEFAULT_FACE_CASCADE.detectMultiScale(im, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
# faces = faces.tolist()
# draw_boxes(im, faces, (0,255,0))
# dresses = dress_boxes(faces)
# draw_boxes(im, dresses, (255,0,0))
# cv2.imshow("faces and dresses", im)
