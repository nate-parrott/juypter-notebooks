# MUST PREFIX CELL WITH %matplotlib inline

import matplotlib.pyplot as plt

def imshow(im):
    plt.figure()
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
# imshow(test_in[0])
# print test_in[0].shape

def imshow_multi(images):
    f = plt.figure()
    for n, im in enumerate(images):
        f.add_subplot(1, len(images), n+1)  # this line outputs images on top of each other
        # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
        fig = plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        fig.axes.get_xaxis().set_visible(False) # this is the worst api in the world
        fig.axes.get_yaxis().set_visible(False)
    plt.show()

imshow_multi(test_in[:10])