'''
Defining different model blocks
Building models will be in different file
blocks include:
- double conv
- downsample block
- upsample block
Building the whole model
'''
from library import *

img_size = 512

#Building blocks of the network
def double_conv_block(x, n_filters):
    x = tf.keras.layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
    return x

def downsample_block(x, n_filters, drop_out):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2)(f)
    p = tf.keras.layers.Dropout(drop_out)(p)
    return f, p

def upsample_block(x,conv_features, n_filters, drop_out):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding='same')(x)
    x = tf.keras.layers.concatenate([x, conv_features])
    x = tf.keras.layers.Dropout(drop_out)(x)
    x = double_conv_block(x, n_filters)
    return x


def build_unet():
    # input layer
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 1))

    # encoder
    f1, block_1 = downsample_block(inputs, 64, drop_out=0.2)
    f2, block_2 = downsample_block(block_1, 128, drop_out=0.3)
    f3, block_3 = downsample_block(block_2, 256, drop_out=0.4)
    f4, block_4 = downsample_block(block_3, 512, drop_out=0.5)

    bottom = double_conv_block(block_4, 1024)

    # decoder
    block_5 = upsample_block(bottom, f4, 512, drop_out=0.5)
    block_6 = upsample_block(block_5, f3, 256, drop_out=0.4)
    block_7 = upsample_block(block_6, f2, 128, drop_out=0.3)
    block_8 = upsample_block(block_7, f1, 64, drop_out=0.2)

    # output layer
    # coloring both background and the target => 2 targets
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(block_8)

    # return the model
    unet_model = tf.keras.Model(inputs, outputs)

    return unet_model


