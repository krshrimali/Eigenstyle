from glob import glob
from random import shuffle
import random
import os
import urllib
import cStringIO
import product_catalog

from PIL import Image
import PIL.ImageOps
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import BallTree
import face_detect

from statistics import mean, median, standard_deviation, inverse_normal_cdf, interquartile_range

LIKE = 'like'
DISLIKE = 'dislike'

N_COMPONENTS = 50
N_COMPONENTS_TO_SHOW = 10
N_DRESSES_TO_SHOW = 5
N_NEW_DRESSES_TO_CREATE = 20

# this is the size of all the Amazon.com images
# If you are using a different source, change the size here 
STANDARD_SIZE = (270, 405)


def open_image_from_url(url):
    return Image.open(cStringIO.StringIO(urllib.urlopen(url).read()))


def img_url_to_array(url):
    """takes a filename and turns it into a numpy array of RGB pixels"""
    img = open_image_from_url(url)
    bbox = img.getbbox()
    if bbox[2] != STANDARD_SIZE[0] or bbox[3] != STANDARD_SIZE[1]:
        img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def img_url_to_array_with_face_detect(url):
    junk, face = face_detect.detect_face(url)
    """takes a filename and turns it into a numpy array of RGB pixels"""
    img = open_image_from_url(url)
    if face is not None:
        dress = face_detect.dress_box(face)
        img = img.crop(dress)
    bbox = img.getbbox()
    if bbox[2] != STANDARD_SIZE[0] or bbox[3] != STANDARD_SIZE[1]:
        img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def img_to_array(filename):
    """takes a filename and turns it into a numpy array of RGB pixels"""
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# write out each eigendress and the dresses that most and least match it
# the file names here are chosen because of the order i wanna look at the results
# (when displayed alphabetically in finder)
def create_eigendress_pictures(raw_data, pca, n_components_to_show=N_COMPONENTS_TO_SHOW,
                               n_dresses_to_show=N_DRESSES_TO_SHOW):
    print("creating eigendress pictures")
    directory = "results/eigendresses/"
    make_folder(directory)
    for i in range(n_components_to_show):
        component = pca.components_[i]
        img = image_from_component_values(component)
        img.save(directory + str(i) + "_eigendress___.png")
        reverse_img = PIL.ImageOps.invert(img)
        reverse_img.save(directory + str(i) + "_eigendress_inverted.png")
        ranked_dresses = sorted(enumerate(X), key=lambda (a, x): x[i])
        most_i = ranked_dresses[-1][0]
        least_i = ranked_dresses[0][0]

        for j in range(n_dresses_to_show):
            most_j = j * -1 - 1
            open_image_from_url(raw_data[ranked_dresses[most_j][0]][2]).save(
                directory + str(i) + "_eigendress__most" + str(j) + ".png")
            open_image_from_url(raw_data[ranked_dresses[j][0]][2]).save(
                directory + str(i) + "_eigendress_least" + str(j) + ".png")


def indexes_for_image_name(raw_data, imageName):
    return [i for (i, (cd, _y, f)) in enumerate(raw_data) if imageName in f]


def predictive_modeling(raw_data, y):
    print("logistic regression...")
    directory = "results/notableDresses/"
    make_folder(directory)

    # split the data into a training set and a test set
    train_split = int(len(raw_data) * 4.0 / 5.0)

    x_train = X[:train_split]
    x_test = X[train_split:]
    y_train = y[:train_split]
    y_test = y[train_split:]

    # if you wanted to use a different model, you'd specify that here
    clf = LogisticRegression(penalty='l2')
    clf.fit(x_train, y_train)

    print "score", clf.score(x_test, y_test)

    # first, let's find the model score for every dress in our dataset
    probs = zip(clf.decision_function(X), raw_data)

    prettiest_liked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == LIKE else 1, p))
    prettiest_disliked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == DISLIKE else 1, p))
    ugliest_liked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == LIKE else 1, -p))
    ugliest_disliked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == DISLIKE else 1, -p))
    in_between_things = sorted(probs, key=lambda (p, (cd, g, f)): abs(p))

    # and let's look at the most and least extreme dresses
    cd = zip(X, raw_data)
    least_extreme_things = sorted(cd, key=lambda (x, (d, g, f)): sum([abs(c) for c in x]))
    most_extreme_things = sorted(cd, key=lambda (x, (d, g, f)): sum([abs(c) for c in x]), reverse=True)

    least_interesting_things = sorted(cd, key=lambda (x, (d, g, f)): max([abs(c) for c in x]))
    most_interesting_things = sorted(cd, key=lambda (x, (d, g, f)): min([abs(c) for c in x]), reverse=True)

    for i in range(10):
        open_image_from_url(prettiest_liked_things[i][1][2]).save(directory + "prettiest_pretty_" + str(i) + ".png")
        open_image_from_url(prettiest_disliked_things[i][1][2]).save(directory + "prettiest_ugly_" + str(i) + ".png")
        open_image_from_url(ugliest_liked_things[i][1][2]).save(directory + "ugliest_pretty_" + str(i) + ".png")
        open_image_from_url(ugliest_disliked_things[i][1][2]).save(
            directory + "directoryugliest_ugly_" + str(i) + ".png")
        open_image_from_url(in_between_things[i][1][2]).save(directory + "neither_pretty_nor_ugly_" + str(i) + ".png")
        open_image_from_url(least_extreme_things[i][1][2]).save(directory + "least_extreme_" + str(i) + ".png")
        open_image_from_url(most_extreme_things[i][1][2]).save(directory + "most_extreme_" + str(i) + ".png")
        open_image_from_url(least_interesting_things[i][1][2]).save(directory + "least_interesting_" + str(i) + ".png")
        open_image_from_url(most_interesting_things[i][1][2]).save(directory + "most_interesting_" + str(i) + ".png")

    # and now let's look at precision-recall
    probs = zip(clf.decision_function(x_test), raw_data[train_split:])
    num_dislikes = len([c for c in y_test if c == 1])
    num_likes = len([c for c in y_test if c == 0])
    lowest_score = round(min([p[0] for p in probs]), 1) - 0.1
    highest_score = round(max([p[0] for p in probs]), 1) + 0.1
    INTERVAL = 0.1

    # first do the likes
    score = lowest_score
    while score <= highest_score:
        true_positives = len([p for p in probs if p[0] <= score and p[1][1] == LIKE])
        false_positives = len([p for p in probs if p[0] <= score and p[1][1] == DISLIKE])
        positives = true_positives + false_positives
        if positives > 0:
            precision = 1.0 * true_positives / positives
            recall = 1.0 * true_positives / num_likes
            print "likes", score, precision, recall
        score += INTERVAL

    # then do the dislikes
    score = highest_score
    while score >= lowest_score:
        true_positives = len([p for p in probs if p[0] >= score and p[1][1] == DISLIKE])
        false_positives = len([p for p in probs if p[0] >= score and p[1][1] == LIKE])
        positives = true_positives + false_positives
        if positives > 0:
            precision = 1.0 * true_positives / positives
            recall = 1.0 * true_positives / num_dislikes
            print "dislikes", score, precision, recall
        score -= INTERVAL

    # now do both
    score = lowest_score
    while score <= highest_score:
        likes = len([p for p in probs if p[0] <= score and p[1][1] == LIKE])
        dislikes = len([p for p in probs if p[0] <= score and p[1][1] == DISLIKE])
        print score, likes, dislikes
        score += INTERVAL


def show_history_of_dress(raw_data, pca, dress_name):
    index = indexes_for_image_name(raw_data, dress_name)[0]
    directory = "results/history/dress" + str(index) + "/"
    make_folder(directory)
    dress = X[index]
    orig_image = raw_data[index][2]
    open_image_from_url(orig_image).save(directory + "dress_" + str(index) + "_original.png")
    for i in range(1, len(dress)):
        reduced = dress[:i]
        construct(pca, reduced, directory + "dress_" + str(index) + "_" + str(i))


def bulk_show_dress_histories(raw_data, pca, lo, hi):
    for index in range(lo, hi):
        directory = "results/history/dress" + str(index) + "/"
        make_folder(directory)
        dress = X[index]
        orig_image = raw_data[index][2]
        open_image_from_url(orig_image).save(directory + "dress_" + str(index) + "_original.png")
        for i in range(1, len(dress)):
            reduced = dress[:i]
            construct(pca, reduced, directory + "dress_" + str(index) + "_" + str(i))


def reconstruct(pca, dress_number, save_name='reconstruct'):
    eigenvalues = X[dress_number]
    construct(pca, eigenvalues, save_name)


def construct(pca, eigenvalues, save_name='reconstruct'):
    components = pca.components_
    eigenzip = zip(eigenvalues, components)
    n = len(components[0])
    r = [int(sum([w * c[i] for (w, c) in eigenzip]))
         for i in range(n)]
    img = image_from_component_values(r)
    img.save(save_name + '.png')


def image_from_component_values(component):
    """takes one of the principal components and turns it into an image"""
    hi = max(component)
    lo = min(component)
    n = len(component) / 3
    divisor = hi - lo
    if divisor == 0:
        divisor = 1

    def rescale(x):
        return int(255 * (x - lo) / divisor)

    d = [(rescale(component[3 * i]),
          rescale(component[3 * i + 1]),
          rescale(component[3 * i + 2])) for i in range(n)]
    im = Image.new('RGB', STANDARD_SIZE)
    im.putdata(d)
    return im


def make_random_dress(pca, save_name, liked):
    random_array = []
    base = likesByComponent if liked else dislikesByComponent
    for c in base[:100]:
        mu = mean(c)
        sigma = standard_deviation(c)
        p = random.uniform(0.0, 1.0)
        num = inverse_normal_cdf(p, mu, sigma)
        random_array.append(num)
    construct(pca, random_array, 'results/createdDresses/' + save_name)


def reconstruct_known_dresses(raw_data, pca):
    print("reconstructing dresses...")
    directory = "results/recreatedDresses/"
    make_folder(directory)
    for i in range(N_DRESSES_TO_SHOW):
        open_image_from_url(raw_data[i][2]).save(directory + str(i) + "_original.png")
        save_name = directory + str(i)
        reconstruct(pca, i, save_name)


def create_new_dresses(pca, n_new_dresses_to_create=N_NEW_DRESSES_TO_CREATE):
    print("creating brand new dresses...")
    directory = "results/createdDresses/"
    make_folder(directory)
    for i in range(n_new_dresses_to_create):
        save_name_like = "newLikeDress" + str(i)
        save_name_dislike = "newDislikeDress" + str(i)
        make_random_dress(pca, save_name_like, True)
        make_random_dress(pca, save_name_dislike, False)


def print_component_statistics_old(n_components_to_show=N_COMPONENTS_TO_SHOW):
    print("component statistics:\n")
    for i in range(n_components_to_show):
        print("component " + str(i) + ":")
        like_comp = likesByComponent[i]
        dislike_comp = dislikesByComponent[i]
        print("means:                     like = " + str(mean(like_comp)) + "     dislike = " + str(mean(dislike_comp)))
        print(
            "medians:                   like = " + str(median(like_comp)) + "     dislike = " + str(median(dislike_comp)))
        print("stdevs:                    like = " + str(standard_deviation(like_comp)) + "     dislike = " + str(
            standard_deviation(dislike_comp)))
        print("interquartile range:       like = " + str(interquartile_range(like_comp)) + "     dislike = " + str(
            interquartile_range(dislike_comp)))
        print("\n")


def print_component_statistics(all_by_components, n_components_to_show=N_COMPONENTS_TO_SHOW):
    print("component statistics:\n")
    for i in range(n_components_to_show):
        print("component " + str(i) + ":")
        comp = all_by_components[i]
        print("means:                     like = " + str(mean(comp)))
        print(
            "medians:                   like = " + str(median(comp)))
        print("stdevs:                    like = " + str(standard_deviation(comp)))
        print("interquartile range:       like = " + str(interquartile_range(comp)))
        print("\n")


def extract_raw_data_from_images(all_urls, process_file):
    print('processing images...')
    print('(this takes a long time if you have a lot of images)')
    raw_data = []
    i = 0
    for url in all_urls:
        try:
            i += 1
            print str(i/float(len(all_urls))), url
            raw_data.append((process_file(url), random.choice([LIKE, DISLIKE]), url))
        except:
            print "process_file failed on", url

    return raw_data


def compute_pca(raw_data):
    # randomly order the data
    # seed(0)
    print('shuffling data...')
    shuffle(raw_data)
    # pull out the features and the labels
    print('pulling out data to run PCA...')
    data = np.array([cd for (cd, _y, f) in raw_data])
    print('finding principal components...')
    pca = RandomizedPCA(n_components=N_COMPONENTS, random_state=0)
    X = pca.fit_transform(data)

    return raw_data, data, pca, X


def random_image_with_neighbors(raw_data, X, tree, k=5):
    make_folder("nearest")
    r = random.randint(0, len(X))
    d, index_array = tree.query(X[r], k=k)
    for i in xrange(0, k):
        open_image_from_url(raw_data[index_array[0][i]][2]).save("nearest/"+str(r)+"_"+str(i)+".png")


def eigenstyle():
    global X
    all_urls = product_catalog.DRESS_URLS
    raw_data = extract_raw_data_from_images(all_urls, img_url_to_array_with_face_detect)
    raw_data, data, pca, X = compute_pca(raw_data)
    create_eigendress_pictures(raw_data, pca, N_COMPONENTS_TO_SHOW, N_DRESSES_TO_SHOW)
    return raw_data, data, pca, X


def neighbors():
    global X
    all_urls = product_catalog.DRESS_URLS
    raw_data = extract_raw_data_from_images(all_urls, img_url_to_array_with_face_detect)
    raw_data, data, pca, X = compute_pca(raw_data)
    return raw_data, data, pca, X, BallTree(X)


# urls = ...
# raw_data = extract_raw_data_from_images(urls, img_url_to_array_with_face_detect)
# raw_data, data, pca, X = compute_pca(raw_data)
# tree = BallTree(X)
# random_image_with_neighbors(raw_data, X, tree, k)
