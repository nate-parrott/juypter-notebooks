import numpy as np

def load_dataset(name):
    images = np.array([image.astype(float) / 255.0 for image in read_images(name + '-images-idx3-ubyte')])
    labels = np.array([label for label in read_labels(name + '-labels-idx1-ubyte')])
    return images, labels

