"""
Build super resolution CNN models.
"""

import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np


def DenseUnit(inputs, filters=16, strides=1, no_layers=8):
    """old implementation"""
    # skips = [inputs]
    # for i in range(no_layers):
    #     if i == 0:
    #         inter_input = inputs
    #     else:
    #         inter_input = keras.layers.Concatenate()(skips)
    #
    #     inter_input = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides,
    #                                       padding="same",
    #                                       activation="relu",
    #                                       kernel_initializer="he_normal",
    #                                       use_bias=True)(inter_input)
    #     skips.append(inter_input)
    #
    # unit_ouput = keras.layers.Concatenate()(skips)

    """new implementation updated 2021-3-31"""
    unit_in = inputs
    for i in range(no_layers):
        x = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides,
                                padding="same",
                                activation="relu",
                                kernel_initializer="he_normal",
                                use_bias=True)(unit_in)
        unit_in = keras.layers.Concatenate()([x, unit_in])

    return unit_in


class ResUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, bn_flag=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.bn_flag = bn_flag
        # self.activation = keras.activations.get(activation)
        self.main_layers = [keras.layers.Conv2D(self.filters, 3, strides=self.strides, padding="same",
                                                kernel_initializer="he_normal", use_bias=False),
                            keras.layers.BatchNormalization(trainable=bn_flag),
                            keras.layers.PReLU(shared_axes=[1, 2]),
                            keras.layers.Conv2D(self.filters, 3, strides=self.strides, padding="same",
                                                kernel_initializer="he_normal", use_bias=False),
                            keras.layers.BatchNormalization(trainable=bn_flag),
                            ]

        self.skip_layers = []
        if self.strides > 1:
            self.skip_layers = [keras.layers.Conv2D(filters=self.filters, kernel_size=1, strides=self.strides,
                                                    padding="same", kernel_initializer="he_normal", use_bias=False)]

    def call(self, inputs, **kwargs):
        output_ = inputs
        for layer in self.main_layers:
            output_ = layer(output_)
        skip_connect = inputs
        for layer in self.skip_layers:
            skip_connect = layer(skip_connect)
        output_ = output_ + skip_connect
        return output_

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "filters": self.filters,
                "strides": self.strides,
                "bn_flag": self.bn_flag}


def SE_layer(inputs, ratio):
    input_shape = inputs.shape
    out_dim = input_shape[-1]
    ave_pool = keras.layers.GlobalAveragePooling2D()(inputs)
    squeeze = keras.layers.Dense(units=int(out_dim // ratio), activation="relu")(ave_pool)
    excitation = keras.layers.Dense(units=out_dim, activation="sigmoid")(squeeze)
    excitation = tf.reshape(excitation, (-1, 1, 1, out_dim))
    se_out = keras.layers.multiply([inputs, excitation])
    return se_out


def blended_attention_unit(inputs, ratio):
    out_dim = inputs.shape[-1]
    squeeze = keras.layers.Conv2D(filters=int(out_dim // ratio), kernel_size=3, strides=1, padding="same",
                                  kernel_initializer="he_normal", activation="relu", use_bias=False)(inputs)
    excitation = keras.layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding="same",
                                     kernel_initializer="he_normal", activation="sigmoid", use_bias=False)(squeeze)
    blended = keras.layers.Multiply()([inputs, excitation])
    return blended


def pyramid_pooling(inputs, pool_scale):
    chan = inputs.shape[-1]
    pool1 = keras.layers.AveragePooling2D(pool_size=(pool_scale, pool_scale))(inputs)
    sp_ext1 = keras.layers.Conv2DTranspose(filters=1, kernel_size=7, strides=pool_scale, kernel_initializer="he_normal",
                                           activation="sigmoid", use_bias=False, padding="same")(pool1)
    spatial_excitation = keras.backend.repeat_elements(sp_ext1, rep=chan, axis=-1)
    sp_ext = keras.layers.Multiply()([inputs, spatial_excitation])
    return sp_ext


def csar_unit(inputs, r, p):
    input_shape = inputs.shape
    chan = input_shape[-1]
    # channel-wise
    ave_pool = keras.layers.GlobalAveragePooling2D()(inputs)
    ave_pool = tf.reshape(ave_pool, (-1, 1, 1, chan))  # after GlobalAvePool, the shape is (batch_size, chan)
    squeeze = keras.layers.Conv2D(filters=int(chan / r), kernel_size=1, strides=1, kernel_initializer="he_normal",
                                  activation="relu", use_bias=False, padding="same")(ave_pool)
    chan_excitation = keras.layers.Conv2D(filters=chan, kernel_size=1, strides=1, kernel_initializer="he_normal",
                                          activation="sigmoid", use_bias=False, padding="same")(squeeze)
    ch_ext = keras.layers.Multiply()([inputs, chan_excitation])

    # spatial-wise
    """old spatial attention"""
    # expansion = keras.layers.Conv2D(filters=chan * p, kernel_size=1, strides=1, kernel_initializer="he_normal",
    #                                 activation="relu", use_bias=False, padding="same")(inputs)
    # sp_ext = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, kernel_initializer="he_normal",
    #                              activation="relu", use_bias=False, padding="same")(expansion)

    """max-average pooling attention"""
    pool = keras.layers.AveragePooling2D(pool_size=(2, 2))(keras.layers.MaxPooling2D(pool_size=(2, 2))(inputs))
    spatial_excitation = keras.layers.Conv2DTranspose(filters=1, kernel_size=7, strides=4,
                                                      kernel_initializer="he_normal",
                                                      activation="sigmoid", use_bias=False, padding="same")(pool)
    spatial_excitation = keras.backend.repeat_elements(spatial_excitation, rep=chan, axis=-1)
    sp_ext = keras.layers.Multiply()([inputs, spatial_excitation])

    """pyramid pooling"""
    # sp_ext = [ch_ext]
    # scale = [2, 4, 8]
    # for s in scale:
    #     sp = pyramid_pooling(inputs, s)
    #     sp_ext.append(sp)

    # channel spatial fusion
    concat = keras.layers.Concatenate()([ch_ext, sp_ext])
    csa = keras.layers.Conv2D(filters=chan, kernel_size=1, strides=1, kernel_initializer="he_normal",
                              activation="relu", use_bias=False, padding="same")(concat)
    # csar = keras.layers.Add()([inputs, csa])

    return csa


# ------------ BaseSRCNN --------
def my_activation(inputs):
    return tf.clip_by_value(inputs, clip_value_min=0.0, clip_value_max=1.0)


def baseSRCNN():
    input_ = keras.layers.Input(shape=(None, None, 1))
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=(9, 9), strides=1, padding="same",
                                activation="relu",
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
                                use_bias=True,
                                bias_initializer="zeros")(input_)

    conv2_1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same",
                                  activation="relu",
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
                                  use_bias=True,
                                  bias_initializer="zeros")(conv1)

    conv3 = keras.layers.Conv2D(filters=1, kernel_size=(5, 5), strides=1, padding="same",
                                activation=my_activation,
                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
                                use_bias=True,
                                bias_initializer="zeros")(conv2_1)

    model = keras.Model(inputs=[input_], outputs=[conv3])
    return model


# -------- Fast-SRCNN -------------
def fsrcnn(d, s, m, upscaling):
    """Build a Fast SRCNN proposed by Dong2016."""
    # (5, d, 1, "patch extraction") -> (1, s, d, "shrinking") -> m * (3, s, s, "mapping") -> (1, d, s, "expanding")
    # -> (9, 1, d, "deconv")

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=d, kernel_size=5, strides=1, padding="same",
                                  use_bias=True, kernel_initializer="he_normal",
                                  input_shape=(None, None, 1), name="feature_extract"))
    model.add(keras.layers.PReLU(shared_axes=[1, 2]))

    model.add(keras.layers.Conv2D(filters=s, kernel_size=1, strides=1, padding="same",
                                  use_bias=True, kernel_initializer="he_normal", name="shrinking"))
    model.add(keras.layers.PReLU(shared_axes=[1, 2]))

    for _ in range(m):
        model.add(keras.layers.Conv2D(filters=s, kernel_size=3, strides=1, padding="same",
                                      use_bias=True, kernel_initializer="he_normal"), )
        model.add(keras.layers.PReLU(shared_axes=[1, 2]))

    model.add(keras.layers.Conv2D(filters=d, kernel_size=1, strides=1, padding="same",
                                  use_bias=True, kernel_initializer="he_normal", name="Expanding"))
    model.add(keras.layers.PReLU(shared_axes=[1, 2]))

    model.add(keras.layers.Conv2DTranspose(filters=1, kernel_size=9, padding="same",
                                           strides=(upscaling, upscaling),
                                           use_bias=True,
                                           kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
                                           name="Deconv"))
    model.build()
    return model


# -------------- Build Dense-Net-SRCNN -------------------
def densenetsr(filters=64, kernel_size=3, strides=1, activation="relu", blocks=8, upscaling=2, attention='', **kwargs):
    """Build a dense-net-SRCNN."""
    default_conv2D = partial(keras.layers.Conv2D,
                             filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             activation=activation,
                             kernel_initializer="he_normal",
                             padding="same")
    bottleneck_layer = partial(keras.layers.Conv2D,
                               filters=256,
                               kernel_size=1,
                               strides=strides,
                               activation=activation,
                               kernel_initializer="he_normal",
                               padding="same")
    default_deconv2D = partial(keras.layers.Conv2DTranspose,
                               filters=filters,
                               kernel_size=3,
                               strides=(upscaling, upscaling),
                               activation=activation,
                               kernel_initializer="he_normal",
                               padding="same")

    input_1 = keras.layers.Input(shape=(None, None, 1))
    # conv1
    conv_1 = default_conv2D(filters=64)(input_1)

    # bottleneck_layer
    # bottleneck = bottleneck_layer()(conv_1)

    # DenseBlocks
    denseblock_skips = [conv_1]
    concat = conv_1
    for i in range(blocks):

        x = DenseUnit(concat, no_layers=6)

        if attention == 'se':
            """ Squeeze and excitation, SE"""
            x = SE_layer(x, ratio=16)
        elif attention == 'blended':
            """ #blended attention block"""
            x = blended_attention_unit(x, 16)
        elif attention == 'csa':
            """ channel-spatial attention"""
            x = csar_unit(x, 16, 2)
        else:
            print('No attention used.')

        # concat = keras.layers.Concatenate()([x, concat])
        concat = x
        denseblock_skips.append(x)

    # concat all dense blocks
    concat = keras.layers.concatenate(denseblock_skips)

    # bottleneck layer for dimension reduction
    bottle_1 = bottleneck_layer(filters=64)(concat)

    #  global feature fusion
    # global_res = keras.layers.Add()([conv_1, bottle_1])

    # deconv_1
    # deconv_1 = default_conv2D(filters=64, kernel_size=5)(bottle_1)
    deconv_1 = default_conv2D(filters=32, kernel_size=3)(bottle_1)

    deconv_1 = keras.activations.relu(tf.nn.depth_to_space(deconv_1, upscaling))

    # reconLayer
    recon = default_conv2D(filters=1)(deconv_1)

    densenet_model = keras.Model(inputs=[input_1], outputs=[recon])
    return densenet_model


# ------- ResNet-SRCNN --------
def pixel_shuffle(I, scalling):
    """The main operation to implement EPSCN"""
    bsize, h, w, ch = tf.shape(I).as_list()
    I = tf.reshape(I, (bsize, h, w, scalling, scalling))  # (bsize, h, w, scalling, scalling)
    I = tf.split(I, h, axis=1)  # h, (bsize, 1, w, scalling, scalling)
    I = [tf.squeeze(x) for x in I]  # h, (bsize, w, scalling, scalling)
    I = tf.concat(I, axis=2)  # (bsize, w, h*scalling, scalling)
    I = tf.split(I, w, axis=1)  # w, (bsize, 1, h*scalling, scalling)
    I = [tf.squeeze(x) for x in I]  # w, (bsize, h*scalling, scalling)
    I = tf.concat(I, axis=2)  # (bsize, h*scalling, w*scalling)
    I = tf.reshape(I, (bsize, h * scalling, w * scalling, 1))
    return I


def resnetsr(filter=64, kernel_sz=3, strides=1, no_resnet=16, upscaling=2):
    """Build ResNet-SR."""
    default_conv2D = partial(keras.layers.Conv2D,
                             filters=filter,
                             kernel_size=kernel_sz,
                             strides=strides,
                             kernel_initializer="he_normal",
                             padding="same")

    input_1 = keras.layers.Input(shape=(None, None, 1))

    conv_1 = default_conv2D(kernel_size=9)(input_1)
    conv_1 = keras.layers.PReLU(shared_axes=[1, 2])(conv_1)

    resblock_ouput = conv_1
    for _ in range(no_resnet):
        resblock_ouput = ResUnit(filters=64)(resblock_ouput)

    conv_2 = default_conv2D()(resblock_ouput)

    bn_1 = keras.layers.BatchNormalization()(conv_2)

    long_skip = keras.layers.add([conv_1, bn_1])

    """Upsampling using transpose_conv2D"""
    # up_1 = keras.layers.Conv2DTranspose(filters=256,
    #                                     kernel_size=kernel_sz,
    #                                     strides=(upscaling, upscaling),
    #                                     kernel_initializer="he_normal",
    #                                     padding="same")(long_skip)
    # up_1 = keras.layers.PReLU(shared_axes=[1, 2])(up_1)

    """Upsampling using ESPCN"""
    conv_3 = default_conv2D(filters=256)(long_skip)
    # p_shuffle = pixel_shuffle(long_skip, scalling)
    p_shuffle_1 = tf.nn.depth_to_space(conv_3, upscaling)
    up_1 = keras.layers.PReLU(shared_axes=[1, 2])(p_shuffle_1)

    # conv_4 = default_conv2D(filters=256)(p_shuffle_1)
    # p_shuffle_2 = tf.nn.depth_to_space(p_shuffle_1, scalling)
    # p_shuffle_2 = keras.layers.PReLU(shared_axes=[1, 2])(p_shuffle_2)

    # recon HR image
    recon = default_conv2D(filters=1, kernel_size=9)(up_1)

    res_model = keras.models.Model(inputs=input_1, outputs=recon)

    return res_model


# SR-GAN network

def disciminator(input_shape, bn_flag=True):
    def dis_block(inputs, filters, strides):
        conv1 = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides,
                                    kernel_initializer="he_normal", padding="same")(inputs)
        bn = keras.layers.BatchNormalization(trainable=bn_flag)(conv1)
        lrelu = keras.activations.relu(bn, alpha=0.2)
        return lrelu

    inputs = keras.layers.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer="he_normal")(inputs)
    lrelu = keras.activations.relu(conv1, alpha=0.2)

    dis_blocks = dis_block(lrelu, 64, 2)

    for i in [2, 4, 8]:
        n_filters = 64 * i
        dis_blocks = dis_block(dis_blocks, n_filters, 1)
        dis_blocks = dis_block(dis_blocks, n_filters, 2)

    flatten = keras.layers.Flatten()(dis_blocks)
    dense1 = keras.layers.Dense(units=1024)(flatten)
    # lrelu2 = keras.activations.relu(dense1, alpha=0.2)
    lrelu2 = keras.layers.ReLU(negative_slope=0.2)(dense1)

    dense2 = keras.layers.Dense(units=1, activation="sigmoid")(lrelu2)

    disc_model = keras.models.Model(inputs=inputs, outputs=dense2)

    disc_model.summary()

    return disc_model

# if __name__ == "__main__":
#     x = np.array(np.random.randn(1, 3, 3, 8)).astype(np.float32)
#     x = csar_unit(x, 2, 2)
#     print(x.shape)
