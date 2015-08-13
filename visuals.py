from PIL import Image
import PIL.ImageOps

from collections import defaultdict
from glob import glob
from random import shuffle, seed
import numpy as np
import pylab as pl
import pandas as pd
import re
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression

N_COMPONENTS = 10

# this will show the 10 prettiest dresses, the 10 ugliest dresses, etc 
# change if you want more or less
N_DRESSES_TO_SHOW = 10

# this is the size of all the Amazon.com images
# If you are using a different source, change the size here 
STANDARD_SIZE = (200,260)
HALF_SIZE = (STANDARD_SIZE[0]/2,STANDARD_SIZE[1]/2)

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

# my files are set up like "images/like/image1.jpg" and "images/dislike/image1.jpg"
like_files = glob('images/like/Image*')
dislike_files = glob('images/dislike/Image*')

process_file = img_to_array

raw_data = [(process_file(filename),'like',filename) for filename in like_files] + \
           [(process_file(filename),'dislike',filename) for filename in dislike_files]

# randomly order the data
seed(0)
shuffle(raw_data)

# pull out the features and the labels
data = np.array([cd for (cd,_y,f) in raw_data])
labels = np.array([_y for (cd,_y,f) in raw_data])

# find the principal components
pca = RandomizedPCA(n_components=N_COMPONENTS, random_state=0)
X = pca.fit_transform(data)
y = [1 if label == 'dislike' else 0 for label in labels]

def image_from_component(component):
    """takes one of the principal components and turns it into an image"""
    hi = max(component)
    lo = min(component)
    n = len(component) / 3
    def rescale(x):
        return int(255 * (x - lo) / (hi - lo))
    d = [(rescale(component[3 * i]),
          rescale(component[3 * i + 1]),
          rescale(component[3 * i + 2])) for i in range(n)]
    im = Image.new('RGB',STANDARD_SIZE)
    im.putdata(d)
    return im

# write out each eigendress and the dresses that most and least match it
# the file names here are chosen because of the order i wanna look at the results
# (when displayed alphabetically in finder)
for i,component in enumerate(pca.components_):
    img = image_from_component(component)
    img.save("results/eigendresses/" + str(i) + "_eigendress___.png")
    reverse_img = PIL.ImageOps.invert(img)
    reverse_img.save("results/eigendresses/" + str(i) + "_eigendress_inverted.png")
    ranked_shirts = sorted(enumerate(X),
           key=lambda (a,x): x[i])
    most_i = ranked_shirts[-1][0]
    least_i = ranked_shirts[0][0]

    for j in range(N_DRESSES_TO_SHOW):
        most_j = j * -1 - 1
        print(most_j)
        Image.open(raw_data[ranked_shirts[most_j][0]][2]).save("results/eigendresses/" + str(i) + "_eigendress__most" + str(j) + ".png")
        Image.open(raw_data[ranked_shirts[j][0]][2]).save("results/eigendresses/" + str(i) + "_eigendress_least" + str(j) + ".png")

def reconstruct(shirt_number):
    """needs 100+ components to look interesting"""
    components = pca.components_
    eigenvalues = X[shirt_number]
    eigenzip = zip(eigenvalues,components)
    N = len(components[0])    
    r = [int(sum([w * c[i] for (w,c) in eigenzip]))
                     for i in range(N)]
    d = [(r[3 * i], r[3 * i + 1], r[3 * i + 2]) for i in range(len(r) / 3)]
    img = Image.new('RGB',STANDARD_SIZE)
    img.putdata(d)
    print raw_data[shirt_number][2]
    img.save('reconstruct.png')

#find and reconstruct the monkey shirt:
#monkey_index = [i for (i,(cd,_y,f)) in enumerate(raw_data) if '243A637' in f]
#reconstruct(282)
    
#
# and now for some predictive modeling

# split the data into a training set and a test set
train_split = int(len(data) * 4.0 / 5.0)

X_train = X[:train_split]
X_test = X[train_split:]
y_train = y[:train_split]
y_test = y[train_split:]

# if you wanted to use a different model, you'd specify that here
clf = LogisticRegression(penalty='l2')
clf.fit(X_train,y_train)

print "score",clf.score(X_test,y_test)
    
# and now some qualitative results

# first, let's find the model score for every dress in our dataset
probs = zip(clf.decision_function(X),raw_data)

prettiest_liked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'like' else 1,p))
prettiest_disliked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'dislike' else 1,p))
ugliest_liked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'like' else 1,-p))
ugliest_disliked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'dislike' else 1,-p))
in_between_things = sorted(probs,key=lambda (p,(cd,g,f)): abs(p))

# and let's look at the most and least extreme dresses
cd = zip(X,raw_data)
least_extreme_things = sorted(cd,key=lambda (x,(d,g,f)): sum([abs(c) for c in x]))
most_extreme_things =  sorted(cd,key=lambda (x,(d,g,f)): sum([abs(c) for c in x]),reverse=True)

least_interesting_things = sorted(cd,key=lambda (x,(d,g,f)): max([abs(c) for c in x]))
most_interesting_things =  sorted(cd,key=lambda (x,(d,g,f)): min([abs(c) for c in x]),reverse=True)

for i in range(10):
    Image.open(prettiest_liked_things[i][1][2]).save("results/notableDresses/prettiest_pretty_" + str(i) + ".png")
    Image.open(prettiest_disliked_things[i][1][2]).save("results/notableDresses/prettiest_ugly_" + str(i) + ".png")
    Image.open(ugliest_liked_things[i][1][2]).save("results/notableDresses/ugliest_pretty_" + str(i) + ".png")
    Image.open(ugliest_disliked_things[i][1][2]).save("results/notableDresses/ugliest_ugly_" + str(i) + ".png")
    Image.open(in_between_things[i][1][2]).save("results/notableDresses/neither_pretty_nor_ugly_" + str(i) + ".png")
    Image.open(least_extreme_things[i][1][2]).save("results/notableDresses/least_extreme_" + str(i) + ".png")
    Image.open(most_extreme_things[i][1][2]).save("results/notableDresses/most_extreme_" + str(i) + ".png")
    Image.open(least_interesting_things[i][1][2]).save("results/notableDresses/least_interesting_" + str(i) + ".png")
    Image.open(most_interesting_things[i][1][2]).save("results/notableDresses/most_interesting_" + str(i) + ".png")


# and now let's look at precision-recall
probs = zip(clf.decision_function(X_test),raw_data[train_split:])
num_dislikes = len([c for c in y_test if c == 1])
num_likes = len([c for c in y_test if c == 0])
lowest_score = round(min([p[0] for p in probs]),1) - 0.1
highest_score = round(max([p[0] for p in probs]),1) + 0.1
INTERVAL = 0.1

# first do the likes
score = lowest_score
while score <= highest_score:
    true_positives  = len([p for p in probs if p[0] <= score and p[1][1] == 'like'])
    false_positives = len([p for p in probs if p[0] <= score and p[1][1] == 'dislike'])
    positives = true_positives + false_positives
    if positives > 0:
        precision = 1.0 * true_positives / positives
        recall = 1.0 * true_positives / num_likes
        print "likes",score,precision,recall
    score += INTERVAL

# then do the dislikes
score = highest_score
while score >= lowest_score:
    true_positives  = len([p for p in probs if p[0] >= score and p[1][1] == 'dislike'])
    false_positives = len([p for p in probs if p[0] >= score and p[1][1] == 'like'])
    positives = true_positives + false_positives
    if positives > 0:
        precision = 1.0 * true_positives / positives
        recall = 1.0 * true_positives / num_dislikes
        print "dislikes",score,precision,recall
    score -= INTERVAL

# now do both
score = lowest_score
while score <= highest_score:
    girls  = len([p for p in probs if p[0] <= score and p[1][1] == 'like'])
    boys = len([p for p in probs if p[0] <= score and p[1][1] == 'dislike'])
    print score, girls, boys
    score += INTERVAL


