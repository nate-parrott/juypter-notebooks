from util import leaky_relu

def create_fc(input, out_size, weight_set=[]):
    # input_dropped = tf.nn.dropout(input, dropout_keep_prob)
    in_size = input.get_shape()[-1]
    w = weight_var([in_size, out_size], stddev=0.1)
    weight_set.append(w)
    b = weight_var([out_size], stddev=0)
    weight_set.append(b)
    x = tf.matmul(input, w)
    return leaky_relu(x + b)

def create_conv(input, out_channels, patch_size=2, stride=2, weight_set=[]):
    in_channels = input.get_shape()[-1]
    w = weight_var([patch_size, patch_size, in_channels, out_channels], stddev=0.1)
    b = weight_var([out_channels], stddev=0)
    conv = tf.nn.conv2d(input, w, strides=[1,stride,stride,1], padding='SAME')
    activation = leaky_relu(conv + b) # tf.nn.relu(conv + b)
    weight_set.append(w)
    weight_set.append(b)
    return activation

def create_deconv(input, out_channels, patch_size=2, stride=2, weight_set=[]):
    # stride should be multiple of patch_size for best results
    in_channels = input.get_shape()[-1]
    input_image_width, input_image_height = input_image_size.get_shape()[2:4]
    
    w = weight_var([patch_size, patch_size, out_channels, in_channels])
    weight_set.append(w)
    b = weight_var([out_channels], init_zero=no_bias)
    weight_set.append(b)
    
    batch_size = tf.shape(input)[0]
    output_shape = tf.pack([batch_size, input_image_width*stride, input_image_height*stride, out_channels])
    
    deconv = tf.nn.conv2d_transpose(input, w, output_shape, strides=[1,stride,stride,1], padding='SAME')
    return leaky_relu(deconv + b)
