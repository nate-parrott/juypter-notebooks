from struct import unpack
import numpy as np
from collections import defaultdict

def read_images(filename):
    data = open(filename).read()
    assert len(data) >= 16
    magic_number, count, rows, cols = unpack('>iiii', data[:16])
    assert magic_number == 2051
    pixels = np.fromstring(data[16:], np.uint8)
    images = pixels.reshape((-1, rows, cols))
    return images

def read_labels(filename):
    data = open(filename).read()
    assert len(data) >= 8
    magic_number, count = unpack('>ii', data[:8])
    assert magic_number == 2049
    labels = np.fromstring(data[8:], np.uint8)
    return labels

def print_image(image):
    for row in image:
        print ''.join([('X' if x else ' ') for x in row])

def load_dataset(name):
    images = np.array([image.astype(float) / 255.0 for image in read_images(name + '-images-idx3-ubyte')])
    labels = np.array([label for label in read_labels(name + '-labels-idx1-ubyte')])
    return images, labels
