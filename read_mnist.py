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

if __name__ == '__main__':
    for dataset in ['train', 't10k']:
        print 'For', dataset, 'dataset:'
        labels = read_labels(dataset + '-labels.idx1-ubyte')
        images = read_images(dataset + '-images.idx3-ubyte')
        for pixel in [(6,6), (13,13)]:
            print ' average value of pixel {0}'.format(pixel)
            x, y = pixel
            counts = defaultdict(int)
            sums = defaultdict(int)
            for label, image in zip(labels, images):
                counts[label] += 1
                sums[label] += image[y][x]
            for n in sorted(counts.keys()):
                avg = sums[n] * 1.0 / counts[n]
                print '  for digit {0}: {1}'.format(n, avg)
            
    # read_labels('train-labels.idx1-ubyte')
    # images = read_images('train-images.idx3-ubyte')
    
