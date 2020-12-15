from lxml import etree
from multiprocessing import Pool, cpu_count
import time
from PIL import Image

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2 as cv


def read_cascade(cascade_path):
    with open(cascade_path) as f:
        xml = f.read()
    root = etree.fromstring(xml)
    cascade = root.find("cascade")
    width = int(cascade.find("width").text)
    height = int(cascade.find("height").text)
    features = cascade.find("features").getchildren()

    # Create feature array
    feature_matrices = np.zeros((len(features), height, width))
    for i, feature in enumerate(features):
        cur_matrix = np.zeros((height, width))
        for rect in feature.find("rects").getchildren():
            line = rect.text.strip().split(" ")
            x1, y1, x2, y2 = map(int, line[:4])
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            c = float(line[4])

            cur_matrix[y1:y2 + 1, x1:x2 + 1] = c

        feature_matrices[i] = cur_matrix

    # Create stages list
    stages = cascade.find("stages")
    stages_list = []
    for stage in stages.getchildren():
        if type(stage) == etree._Element:
            threshold = float(stage.find("stageThreshold").text)
            clfs = stage.find("weakClassifiers")

            classifiers = []
            for clf in clfs:
                internal_nodes = clf.find("internalNodes").text.strip().split(" ")
                feature_num = int(internal_nodes[2])
                feature_thresh = float(internal_nodes[3])

                leafs = clf.find("leafValues").text.strip().split(" ")
                less_leaf = float(leafs[0])
                greater_leaf = float(leafs[1])

                classifiers.append([feature_num, feature_thresh, less_leaf, greater_leaf])

            stages_list.append([threshold, classifiers])

    return width, height, feature_matrices, stages_list


def show_points_of_interest(photo_path, width, height, feature_matrices, stages_list):
    image = cv.imread(photo_path, 0)
    scale_percent = 40
    _w = int(image.shape[1] * scale_percent / 100)
    _h = int(image.shape[0] * scale_percent / 100)
    dim = (_w, _h)
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    image_height, image_width = image.shape[0], image.shape[1]
    plt.imshow(image)
    count = 0
    for stage in stages_list:
        count += 1
        image_copy = image.copy()

        for classifier in stage[1]:
            feature_num, thresh, less, greater = classifier

            activation_map = convolve2d(image, feature_matrices[feature_num], mode="valid")

            if greater > less:
                activation_map[activation_map < thresh] = 0
            else:
                activation_map[activation_map > thresh] = 0

            # top 5 non-zero activations
            k = 5
            flatten_activation_map = activation_map.flatten()
            top_indices = np.argpartition(flatten_activation_map, -k)[-k:]

            # filter zero activations
            top_indices = top_indices[flatten_activation_map[top_indices] > 0]

            for top_index in top_indices:
                i, j = np.unravel_index(top_index, activation_map.shape)

                image_part = image[i:i + height, j:j + width].astype(np.uint8)
                rectangle = np.ones(image_part.shape, dtype=np.uint8) * 255

                res = cv.addWeighted(image_part, 0.5, rectangle, 0.5, 1.0)
                image_copy[i:i + height, j:j + width] = res

        plt.figure()
        plt.imshow(image_copy, cmap="gray")
        plt.show()

        im = Image.fromarray(image_copy)
        im.save('points\\' + str(count)+'.jpg')

def show_points(photo_path, width, height, feature_matrices, stages_list, number):
    image = cv.imread(photo_path, 0)
    scale_percent = 40
    _w = int(image.shape[1] * scale_percent / 100)
    _h = int(image.shape[0] * scale_percent / 100)
    dim = (_w, _h)
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    image_height, image_width = image.shape[0], image.shape[1]
    stage = stages_list[number]
    image_copy = image.copy()

    for classifier in stage[1]:
        feature_num, thresh, less, greater = classifier

        activation_map = convolve2d(image, feature_matrices[feature_num], mode="valid")

        if greater > less:
            activation_map[activation_map < thresh] = 0
        else:
            activation_map[activation_map > thresh] = 0

        # top 5 non-zero activations
        k = 5
        flatten_activation_map = activation_map.flatten()
        top_indices = np.argpartition(flatten_activation_map, -k)[-k:]

        # filter zero activations
        top_indices = top_indices[flatten_activation_map[top_indices] > 0]

        for top_index in top_indices:
            i, j = np.unravel_index(top_index, activation_map.shape)

            image_part = image[i:i + height, j:j + width].astype(np.uint8)
            rectangle = np.ones(image_part.shape, dtype=np.uint8) * 255

            res = cv.addWeighted(image_part, 0.5, rectangle, 0.5, 1.0)
            image_copy[i:i + height, j:j + width] = res

        im = Image.fromarray(image_copy)
        im.save('tmp\\' + str(number) + '.jpg')


def show_feature(features, i):
    plt.imshow(features[i], cmap="gray")
    plt.show()
    plt.imsave('test.png', features[i], cmap="gray")



if __name__ == '__main__':
    cascade_path = 'source\cascade_cat.xml'
    #cascade_glitch = 'updated.xml'
    w, h, f, s = read_cascade(cascade_path)
    #show_points_of_interest('cats/7.jpg', w, h, f, s)
    show_feature(f, 1)
