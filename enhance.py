import cv2
from math import floor, ceil, log
import os
import face_detect
import colorcorrect.algorithm
import re
import product_catalog
import pymongo
import subprocess
import numpy as np

BRISQUE = 'brisque_revised/brisquequality'
BRISQUE_SCORE_PATTERN = re.compile('score in main file is given by:(-?\\d+\\.\\d+)')


def brisque_quality_score(image_path):
    process = subprocess.Popen([BRISQUE, '-im', image_path], stdout=subprocess.PIPE)
    process_output = process.communicate()[0]
    match = BRISQUE_SCORE_PATTERN.search(process_output)
    if match is None or not len(match.groups()) == 1:
        print "{}: BRISQUE output was: {}".format(image_path, process_output)
        return 100.0, 0.0
    # otherwise we're good so return score
    brisque = float(match.groups()[0])
    score = brisque
    score = max(0.0, min(100.0, score))  # clip to [0,100]
    score = 100.0 - score  # make higher better instead of lower
    return brisque, score/100.0


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


def auto_crop_hsv(src, crop_white=False):
    try:
        if face_detect.is_color(src):
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
        else:
            v = src
        if crop_white:
            v = abs(255-v)  # invert light to dark
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
    except:
        return src


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
    h_ = min(im_height-bottom_of_face, h*9)  # bottom of face to bottom of image or to a multiple of face height
    w_ = int(w*4.5)  # based on face width
    x_ = x+(w/2)-(w_/2)  # centered below face
    return x_, y_, w_, h_


def crop_to_human(im, faces, dresses):
    dress_bounds = bounding_box(dresses)
    # preserve ~1 face height above top face and crop off the rest
    top_face_y = min([f[1] for f in faces])
    max_face_height = max([f[3] for f in faces])
    y1 = max(0, top_face_y - max_face_height)
    y2 = min(im.shape[0] - 1, dress_bounds[1]+dress_bounds[3])
    # preserve roughly the area horizontally covered by the people's bodies (approximate width below each face)
    x1 = dress_bounds[0]
    x2 = dress_bounds[2] + x1
    x1 = max(0, x1)
    x2 = min(im.shape[1] - 1, x2)
    im = im[y1:y2 + 1, x1:x2 + 1]
    return im


def face_score(faces):
    if len(faces) == 0:
        score = 0.5  # same as 3 faces
    elif len(faces) == 1:
        score = 1.0
    else:
        score = 1.0 / (len(faces)-1)
    return len(faces), score


def fit_in(src, max_h, max_w):
    h, w = src.shape[:2]
    if h <= max_h and w <= max_w:
        return src
    smoosh = min(max_h/float(h), max_w/float(w))
    dimensions = (int(w*smoosh), int(h*smoosh))  # yes, amazingly width is first in this case... ugh.
    print('smooshing {} {} => {} {}'.format(h, w, dimensions[1], dimensions[0]))
    return cv2.resize(src, dimensions, interpolation=cv2.INTER_AREA)


def adaptive_histogram_equalize(src):
    is_color = face_detect.is_color(src)
    if is_color:
        ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(ycrcb)
    else:
        y = src
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    dst = clahe.apply(y)
    # dst = cv2.equalizeHist(y)
    if is_color:
        ycrcb = cv2.merge((dst, cr, cb))
        dst = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return dst


def grabCut(src, box):
    mask = np.zeros(src.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(src, mask, box, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    dst = src*mask2[:, :, np.newaxis]
    return dst


def auto_fix(im, noise_removal=False):
    im = auto_crop_hsv(im)
    im = auto_crop_hsv(im, crop_white=True)
    min_face_size = (int(im.shape[0]*0.05), int(im.shape[1]*0.05))
    faces = face_detect.detect_faces(im, min_neighbors=5, min_size=min_face_size, max_size=None)
    # keep faces above the fold
    faces = [face for face in faces if face[1] < im.shape[0]*0.4]
    # find dresses
    dresses = [dress_box2(face, im.shape[:2]) for face in faces]
    # if len(dresses) > 0:
    #     print('grabcut!')
    #     im = grabCut(im, bounding_box(dresses+faces))
    if len(faces) > 0:
        im = crop_to_human(im, faces, dresses)
    # limit max size (after cropping)
    im = fit_in(im, 1800, 1200)
    if noise_removal:
        im = cv2.fastNlMeansDenoisingColored(im)
    im = face_detect.skin_detect2(im, marks=True)
    # im = simplest_color_balance(im)
    # print('retinex starting...')
    # im = colorcorrect.algorithm.retinex_with_adjust(im)
    # print('retinex complete.')
    #face_detect.draw_boxes(im, faces)
    #face_detect.draw_boxes(im, dresses, (255, 0, 0))
    # face_detect.draw_boxes(im, people, (255, 0, 0))
    return im, faces


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
            im, faces = auto_fix(im)
            outfile = os.path.splitext(file)[0]+"_fixed.png"
            cv2.imwrite(os.path.join(dir, outfile), im)


def brisque_quality_score_from_memory(fixed):
    temp_path = 'temp.png'
    cv2.imwrite(temp_path, fixed)
    brisque = brisque_quality_score(temp_path)
    os.remove(temp_path)
    return brisque


def resolution_score(size):
    l = log(size[0]*size[1])  # range ends up being about 10-16
    l = l*10.0  # about 100-170
    l = max(0.0, l-100.0)  # clip to min
    l = min(70.0, l)  # clip to max
    l = l/70.0  # scale to 0-1
    return size[0], size[1], l


def lightness_score(im):
    if face_detect.is_color(im):
        hls = cv2.cvtColor(im, cv2.cv.CV_BGR2HLS)
        return hls[:, :, 1].mean()/255.0
    return im.mean()/255.0


def lightness_score_normal(im, mean=0.5):
    # actual mean lightness of data is ~0.4, but I want to favor brighter images so I'm using 0.5
    # 2.71828^(-37.5455 (x-mean)^2)  # normal distro with stdev 0.1154
    e = 2.71828
    im_mean = lightness_score(im)
    diff = im_mean-mean
    return im_mean, e**(-37.5455*diff*diff)


def fix_all_from_mongo_and_update(remote_update=False, local_output=False, first_style_index=0, first_image_index=0,
                                  style_limit=99999, images_per_style=99999):
    db = product_catalog.connect_to_mongo()
    cursor = db.Review.find({"status": "pending", 'photoIds': {'$exists': True}}, {'styleName': 1})
    pending_styles = {s['styleName']: True for s in cursor}
    pending_dress_styles = [s for s in pending_styles if s in product_catalog.DRESS_STYLES]
    # sort styles with fewest reviews first
    cursor = db.Review.aggregate([{'$group': {'_id': "$styleName", 'count': {'$sum': 1}, 'avg': {'$avg': "$rating"}}}])
    sort_keys = {c['_id']: (c['count'], c['avg']) for c in cursor}
    pending_dress_styles.sort(key=lambda x: sort_keys[x])

    style_to_image_list = []
    print('pending dress styles {}'.format(len(pending_dress_styles)))
    for i, style in enumerate(pending_dress_styles):
        if i == style_limit:
            break
        print('{:4}'.format(i))
        # cursor = db.Photo.find({'status': 'approved', 'styleName': style}, {'_id': 1, 'styleName': 1, 'relUrls': 1})
        cursor = db.Photo.find({'status': 'pending', 'styleName': style, 'deletedAt': {'$exists': False}, 'relUrls': {'$exists': True}},
                               {'_id': 1, 'styleName': 1, 'relUrls': 1}).limit(images_per_style)  # .sort('_created', pymongo.DESCENDING)
        try:
            image_paths = map(product_catalog.get_id_and_original_image_url, cursor)
            if len(image_paths) > 0:
                style_to_image_list.append((style, image_paths))
        except KeyError:
            pass
    print('style count: {}'.format(len(style_to_image_list)))

    if local_output and not os.path.exists('review_im'):
        os.mkdir('review_im')
    for s, style_to_image in enumerate(style_to_image_list):
        style, image_paths = style_to_image
        if s < first_style_index:
            continue
        print('{:4} style {} image count {}'.format(s, style, len(image_paths)))
        style_dir = os.path.join('review_im', style)
        if local_output and not os.path.exists(style_dir):
            os.mkdir(style_dir)
        for i, id_and_path in enumerate(image_paths):
            if s == first_style_index and i < first_image_index:
                continue
            try:
                oid, path = id_and_path
                print('{:3} {} fixing image...'.format(i, path))
                im = face_detect.cv_open_image_from_url(path)
                fixed, faces = auto_fix(im)
                print('fixed dimensions: {:4} {:4} {:,}'.format(fixed.shape[0], fixed.shape[1], fixed.shape[0]*fixed.shape[1]))
                if remote_update:
                    print("     uploading fixed image...")
                    temp_file = 'fixed.png'
                    cv2.imwrite(temp_file, fixed)
                    upload_result = product_catalog.put_preprocessed_image(oid, temp_file)
                    os.remove(temp_file)
                print("     scoring...")
                brisque, bscore = brisque_quality_score_from_memory(fixed)
                print("     brisque score:    {:01.2f}".format(bscore))
                face_count, fscore = face_score(faces)
                print("     face score:       {:01.2f} {}".format(fscore, face_count))
                h, w, rscore = resolution_score(fixed.shape[:2])
                print("     resolution score: {:01.2f}".format(rscore))
                lightness, lscore = lightness_score_normal(fixed)
                print("     lightness score:  {:01.2f} {:01.2f}".format(lscore, lightness))
                # compute blended score
                score = bscore * fscore * rscore * lscore
                print("     ********** score: {:01.2f}".format(score))
                if remote_update:
                    print("     updating mongo...")
                    db.Photo.update(
                        {'_id': oid},
                        {
                            '$set':
                            {
                                'relUrls.imageCroppedThumb': upload_result['IMAGE_THUMB'],
                                'relUrls.imageCroppedOriginal': upload_result['IMAGE_ORIGINAL'],
                                'relUrls.imageCropped': upload_result['IMAGE'],
                                'relUrls.imageCroppedLargeThumb': upload_result['IMAGE_LARGE_THUMB'],
                                'analysis.brisque': brisque,
                                'analysis.brisqueScore': bscore,
                                'analysis.faceCount': face_count,
                                'analysis.faceScore': fscore,
                                'analysis.height': h,
                                'analysis.width': w,
                                'analysis.resolutionScore': rscore,
                                'analysis.lightnessMean': lightness,
                                'analysis.lightnessScore': lscore,
                                'analysis.compositeScore': score,
                                'analysis.revision': 42
                            }
                        }
                    )
                if local_output:
                    cv2.imwrite(os.path.join(style_dir, '{:04}_{:03}__.png'.format(int(score*1000), i)), im)
                    cv2.imwrite(os.path.join(style_dir, '{:04}_{:03}_fixed.png'.format(int(score*1000), i)), fixed)
            except:
                pass


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

# image 14 of style 10
