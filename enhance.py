import cv2
from math import floor, ceil, log
import os
import face_detect
from colorcorrect.algorithm import automatic_color_equalization
import subprocess
import re
import product_catalog


BRISQUE = 'brisque_revised/brisquequality'
BRISQUE_SCORE_PATTERN = re.compile('score in main file is given by:(-?\\d+\\.\\d+)')


def brisque_quality_score(image_path):
    process = subprocess.Popen([BRISQUE, '-im', image_path], stdout=subprocess.PIPE)
    process_output = process.communicate()[0]
    match = BRISQUE_SCORE_PATTERN.search(process_output)
    if match is None or not len(match.groups()) == 1:
        print "{}: BRISQUE output was: {}".format(image_path, process_output)
        return None
    # otherwise we're good so return score
    score = float(match.groups()[0])
    score = max(0.0, min(100.0, score))  # clip to [0,100]
    score = 100.0 - score  # make higher better instead of lower
    return score/100.0


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
        if is_color(src):
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


def is_color(im):
    return im is not None and im.shape is not None and len(im.shape) == 3 and im.shape[2] == 3


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
    print('{} {}'.format(im.shape[0] - 1, dress_bounds[1]+dress_bounds[3]))
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
    return score


def auto_fix(im):
    im = auto_crop_hsv(im)
    im = auto_crop_hsv(im, crop_white=True)
    min_face_size = (int(im.shape[0]*0.05), int(im.shape[1]*0.05))
    faces = face_detect.detect_faces(im, min_neighbors=5, min_size=min_face_size, max_size=None)
    # keep faces above the fold
    faces = [face for face in faces if face[1] < im.shape[0]*0.4]
    # find dresses
    dresses = [dress_box2(face, im.shape[:2]) for face in faces]
    if len(faces) > 0:
        im = crop_to_human(im, faces, dresses)
    #im = cv2.fastNlMeansDenoisingColored(im)
    #im = simplest_color_balance(im)
    #face_detect.draw_boxes(im, faces)
    #face_detect.draw_boxes(im, dresses, (255, 0, 0))
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
    return l


def lightness_score(im):
    hls = cv2.cvtColor(im, cv2.cv.CV_BGR2HLS)
    return hls[:, :, 1].mean()/255.0


def fix_some_from_mongo(style_limit=10, images_per_style=10):
    db = product_catalog.connect_to_mongo()
    style_to_image_map = {}
    for i, style in enumerate(product_catalog.DRESS_STYLES.keys()):
        if i == style_limit:
            break
        cursor = db.Photo.find({'status': 'approved', 'styleName': style}, {'_id': 1, 'styleName': 1, 'relUrls': 1}).limit(images_per_style)
        image_paths = map(product_catalog.get_id_and_original_image_url, cursor)
        if len(image_paths) > 0:
            style_to_image_map[style] = image_paths
    print('style count: {}'.format(len(style_to_image_map)))

    if not os.path.exists('review_im'):
        os.mkdir('review_im')
    for style in style_to_image_map.keys():
        print('style {} image count {}'.format(style, len(style_to_image_map[style])))
        style_dir = os.path.join('review_im', style)
        if not os.path.exists(style_dir):
            os.mkdir(style_dir)
        for i, id_and_path in enumerate(style_to_image_map[style]):
            oid, path = id_and_path
            print('{:3} {}'.format(i, path))
            im = face_detect.cv_open_image_from_url(path)
            fixed, faces = auto_fix(im)
            bscore = brisque_quality_score_from_memory(fixed)
            print("brisque score:    {0}".format(bscore))
            fscore = face_score(faces)
            print("face score:       {0}".format(fscore))
            rscore = resolution_score(fixed.shape[:2])
            print("resolution score: {0}".format(rscore))
            # compute blended score
            score = int(0.3333*bscore + 0.3333*fscore + 0.3333*rscore)
            print("********** score: {0}".format(score))
            db.Photo.update(
                {'_id': oid},
                {
                    '$set':
                    {
                        'analysis.brisqueScore': bscore,
                        'analysis.faceScore': fscore,
                        'analysis.resolutionScore': rscore,
                        'analysis.compositeScore': score
                    }
                }
            )
            cv2.imwrite(os.path.join(style_dir, '{:02}_{:03}__.png'.format(score, i)), im)
            cv2.imwrite(os.path.join(style_dir, '{:02}_{:03}_fixed.png'.format(score, i)), fixed)


def fix_all_from_mongo_and_update(first_style_index=0, first_image_index=0):
    db = product_catalog.connect_to_mongo()
    style_to_image_map = {}
    for i, style in enumerate(product_catalog.DRESS_STYLES.keys()):
        print('{:4}'.format(i))
        cursor = db.Photo.find({'status': 'approved', 'styleName': style}, {'_id': 1, 'styleName': 1, 'relUrls': 1})
        try:
            image_paths = map(product_catalog.get_id_and_original_image_url, cursor)
            if len(image_paths) > 0:
                style_to_image_map[style] = image_paths
        except KeyError:
            pass
    print('style count: {}'.format(len(style_to_image_map)))

    for s, style in enumerate(style_to_image_map.keys()):
        if s < first_style_index:
            continue
        print('{:4} style {} image count {}'.format(s, style, len(style_to_image_map[style])))
        for i, id_and_path in enumerate(style_to_image_map[style]):
            if s == first_style_index and i < first_image_index:
                continue
            try:
                oid, path = id_and_path
                print('{:3} {}'.format(i, path))
                im = face_detect.cv_open_image_from_url(path)
                fixed, faces = auto_fix(im)
                bscore = brisque_quality_score_from_memory(fixed)
                print("brisque score:    {:01.2f}".format(bscore))
                fscore = face_score(faces)
                print("face score:       {:01.2f}".format(fscore))
                rscore = resolution_score(fixed.shape[:2])
                print("resolution score: {:01.2f}".format(rscore))
                lscore = lightness_score(fixed)
                print("lightness score:  {:01.2f}".format(lscore))
                # compute blended score
                score = bscore * fscore * rscore * lscore
                print("********** score: {:01.2f}".format(score))
                db.Photo.update(
                    {'_id': oid},
                    {
                        '$set':
                        {
                            'analysis.brisqueScore': bscore,
                            'analysis.faceScore': fscore,
                            'analysis.resolutionScore': rscore,
                            'analysis.lightnessScore': lscore,
                            'analysis.compositeScore': score
                        }
                    }
                )
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
