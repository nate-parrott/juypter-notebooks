import numpy as np
import tensorflow as tf
from PIL import Image

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

def rand_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def load_image(filename, size):
    pic = Image.open(filename)
    pic.thumbnail((size, size), Image.ANTIALIAS)
    
    background = Image.new("RGB", pic.size, (255, 255, 255))
    background.paste(pic, mask=pic.split()[3]) # 3 is the alpha channel
    pix = np.array(background.getdata()).reshape(pic.size[0], pic.size[1], 3) / 255.0
    return pix

def weight_var(shape, init_zero=False, stddev=0.1):
    if init_zero: stddev = 0
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def show_image(arr, path=None):
    im = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), 'RGB')
    if path:
        im.save(path)
    else:
        im.show()

def show_image_grayscale(arr, path=None):
    if len(arr.shape) == 3:
        assert arr.shape[-1] == 1
        new_shape = (arr.shape[0], arr.shape[1])
        arr = arr.reshape(new_shape)
    im = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), 'L')
    if path:
        im.save(path)
    else:
        im.show()

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

def leaky_relu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)
