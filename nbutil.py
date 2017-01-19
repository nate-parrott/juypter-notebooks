# MUST PREFIX CELL WITH %matplotlib inline if using imshow*

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def imshow(im):
    plt.figure()
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
# imshow(test_in[0])
# print test_in[0].shape

def imshow_multi(images):
    f = plt.figure()
    for n, im in enumerate(images):
        im = np.clip(im, 0, 1)
        f.add_subplot(1, len(images), n+1)  # this line outputs images on top of each other
        # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
        fig = plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        fig.axes.get_xaxis().set_visible(False) # this is the worst api in the world
        fig.axes.get_yaxis().set_visible(False)
    plt.show()

# imshow_multi(test_in[:10])

def to_pil(img):
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
