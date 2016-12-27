import numpy as np
import tensorflow as tf

import vgg16
import utils

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    
    images = tf.placeholder("float", [None, 224, 224, 3])

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    
    while True:
        url = raw_input(" URL: ")
        if url:
            img = utils.load_image(url)
            batch = img.reshape((1, 224, 224, 3))
            prob = sess.run(vgg.prob, feed_dict={images: batch})[0]
            utils.print_prob(prob, './synset.txt')
