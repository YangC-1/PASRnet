"""
data processing functions
"""
import os
import cv2
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from models.srnns import disciminator, resnetsr, ResUnit
from tqdm import tqdm
from tensorflow.keras.applications.vgg19 import VGG19
import pandas as pd
import logging
from models.srnns import *
from models.FD_UNet import getModel
from models.rdn import RDN
from models.FD_UNet import getModel
from models import Res_UNet
from sklearn.decomposition import PCA


#  functions

# ---------- preprocessing functions --------------


def generate_pca(ksize=7, dim_pca=0.99, num_kernels=5000):
    kernels = np.zeros((num_kernels, ksize ** 2), dtype=np.float32)
    for i in range(num_kernels):
        sigma = np.around(0.5 + (3 - 0.5) * np.random.random(), decimals=2)
        gauss_kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
        gauss_kernel = gauss_kernel * np.transpose(gauss_kernel, (1, 0))
        kernels[i, :] = gauss_kernel.flatten()

    # get PCA components
    pca = PCA(n_components=dim_pca).fit(kernels).components_
    return pca


def load_train_data(data_path):
    logging.info('loading train data...')
    data = np.load(data_path)
    logging.info('Size of train data: ({}, {}, {})'.format(data.shape[0], data.shape[1], data.shape[2]))
    return data


def train_datagen(y_, pca, down_scale=2, batch_size=16, unet=False, blind=False):
    assert y_.shape[0] % down_scale == 0 and y_.shape[1] % down_scale == 0, \
        " Input shape must be integer times of down_scale"

    indices = list(range(y_.shape[0]))
    while True:
        np.random.seed(seed=0)
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_y = y_[indices[i:i + batch_size]]

            # blur batches
            batch_y_ = np.transpose(np.squeeze(batch_y), (1, 2, 0))
            # batch_y_ = np.reshape(batch_y_, (y.shape[1], y.shape[2], y.shape[0]))     # the results are not correct
            if blind:
                sigma = np.around(0.5 + (3 - 0.5) * np.random.rand(), decimals=2)
                blur_batch_x_ = cv2.GaussianBlur(batch_y_, ksize=(7, 7), sigmaX=sigma, sigmaY=sigma,
                                                 borderType=cv2.BORDER_REPLICATE)
            else:
                blur_batch_x_ = cv2.GaussianBlur(batch_y_, ksize=(7, 7), sigmaX=1.6, sigmaY=1.6,
                                                 borderType=cv2.BORDER_REPLICATE)

            blur_batch_x = np.transpose(blur_batch_x_, (2, 0, 1))
            blur_batch_x = blur_batch_x[:, :, :, np.newaxis]

            if unet:
                mask_x = np.asarray(
                    list(map(lambda x: 1 if x % down_scale == 0 else 0, list(range(blur_batch_x.shape[1])))))
                mask_y = np.asarray(
                    list(map(lambda x: 1 if x % down_scale == 0 else 0, list(range(blur_batch_x.shape[2])))))
                downsample_blur_batch_x = blur_batch_x * mask_x[:, np.newaxis, np.newaxis] * mask_y[:, np.newaxis]
            else:
                # down_sampling by down_scale
                downsample_blur_batch_x = blur_batch_x[:, ::down_scale, 0::down_scale]

            if pca.any():
                # pca = generate_pca()
                gauss_kernel = cv2.getGaussianKernel(ksize=7, sigma=sigma)
                gauss_kernel = gauss_kernel * np.transpose(gauss_kernel, (1, 0))
                degradation_vector = np.dot(pca, gauss_kernel.flatten())
                degradation_map = degradation_vector[np.newaxis, np.newaxis, np.newaxis, :] * np.ones((downsample_blur_batch_x.shape[0], downsample_blur_batch_x.shape[1], downsample_blur_batch_x.shape[2], 1))
                downsample_blur_batch_x = np.concatenate((downsample_blur_batch_x, degradation_map), axis=-1)

            yield downsample_blur_batch_x, batch_y


def generate_train(datafolder, savepath, size_input, size_label, scale, stride):
    data_x, label_y = [], []
    cnt = 0
    for filename in os.listdir(datafolder):

        # -------- load image using cv2 -----------
        # img = cv2.imread(os.path.join(datafolder, filename))
        # if img.shape[2] == 3:
        #     img_ycbcr = rgb2ycbcr(img)
        #     img = img_ycbcr[:, :, 0]/255.0
        #
        # # generate hr and lr image
        # img_label = modcrop(img, scale)
        # hr_shape = img_label.shape                  # (height, width)
        # lr_shape = (int(hr_shape[0] / scale), int(hr_shape[1] / scale))
        # img_input = cv2.resize(img_label, lr_shape[::-1], interpolation=cv2.INTER_CUBIC)  # cv2.resize expect the dsize of (width, height)

        # -------- load image using PIL ------------

        img = Image.open(os.path.join(datafolder, filename))
        if img.mode == "RGB":
            img = img.convert("YCbCr")
            img = np.asarray(img)[:, :, 0]

        img_label = modcrop(img, scale)
        hr_shape = img_label.shape
        lr_shape = [int(x / scale) for x in hr_shape]

        img_label = Image.fromarray(img_label)
        img_input = np.asarray(img_label.resize(lr_shape[::-1], Image.BICUBIC)) / 255.0
        img_label = np.asarray(img_label) / 255.0

        # generate batches
        for x in range(0, lr_shape[0] - size_input, stride):
            for y in range(0, lr_shape[1] - size_input, stride):
                locx = scale * x
                locy = scale * y

                subim_input = img_input[x:x + size_input, y:y + size_input]
                subim_label = img_label[locx:locx + size_label, locy:locy + size_label]

                data_x.append(subim_input[:, :, np.newaxis])
                label_y.append(subim_label[:, :, np.newaxis])
                cnt += 1

    data = np.asarray(data_x)
    del data_x
    label = np.asarray(label_y)
    del label_y

    order = list(range(cnt))
    random.shuffle(order)
    data = data[order, :, :, :]
    label = label[order, :, :, :]

    # write to HDF5 file
    with h5py.File(savepath, "w") as f:
        f.create_dataset("data", data=data, dtype=np.float32, chunks=(32, size_input, size_input, 1))
        f.create_dataset("label", data=label, dtype=np.float32, chunks=(32, size_label, size_label, 1))


def rgb2ycbcr(img):
    im_rgb = img.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]
    return im_ycbcr


def modcrop(img, scale):
    img_shape = np.asarray(img.shape)
    img_shape[0:2] = img_shape[0:2] - img_shape[0:2] % scale
    crop_img = img[0:img_shape[0], 0:img_shape[1]]
    return crop_img


def to_tensor(x):
    if x.ndim == 2:
        x = x[np.newaxis, ..., np.newaxis]
    elif x.ndim == 3:
        x = x[np.newaxis, ...]

    return x


def from_tensor(x):
    return np.squeeze(x)


# ---------- image show and save -------------

def save_result(x, path):
    if path.endswith('.jpg') or path.endswith('.png') or path.endswith('.bmp') or path.endswith(
            '.tif') or path.endswith('.jpeg'):
        x = x - np.min(x)
        x = x / np.max(x) * 255.0
        cv2.imwrite(path, x.astype('uint8'))
    elif path.endswith('.txt'):
        np.savetxt(path, x, fmt='%2.4f')
    else:
        logging.warning('Unsupported saving type.')


# ---------- custum metrics --------------


def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def psnr(y_true, y_pred):
    # assert y_true.shape[1:] == y_pred.shape[1:], "Cannot calculate PSNR. Input shapes not same." \
    #                                      " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
    #                                                                                 str(y_pred.shape))
    # snr = -10 * tf.math.log(tf.reduce_mean(tf.square(y_pred - y_true)))
    snr = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return snr


# ---------- custum losses --------------
class VGG_loss:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        v19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        v19.trainable = False

        for l in v19.layers:
            l.trainable = False
        model = keras.models.Model(inputs=v19.input, outputs=v19.get_layer('block5_conv4').output)
        model.trainable = False

        self.model = model

    def vgg_loss(self, y_true, y_pred):
        return keras.backend.mean(keras.backend.square(self.model(y_true) - self.model(y_pred)))


class TV_loss:
    def __init__(self, p):
        self.p = p

    def tv_loss(self, y_true, y_pred):
        h, w, c = y_true.shape
        indx = list(range(1, h)) + [h - 1]

        fx = y_true[indx, ::] - y_true
        fy = y_true[:, indx, :] - y_true

        tv_loss = self.p * np.sum(np.sqrt(np.square(fx) + np.square(fy)))
        mse_loss = np.mean(np.square(y_true - y_pred))

        return mse_loss + tv_loss


# ---------- training functions --------------
def gan_model(discriminator, generator, shape, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = keras.layers.Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = keras.models.Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(optimizer=optimizer,
                loss=[vgg_loss, "binary_crossentropy"], loss_weights=[1.0, 1e-3],
                metrics={"functional_1": ssim, "functional_3": "acc"})

    gan.summary()

    return gan


def train_srgan(n_epochs, dataset, train_ds_size, batch_size, optimizer, scale, image_shape, pretrain: bool, savepath):
    loss = VGG_loss(image_shape)
    input_shape = (image_shape[0] // scale, image_shape[1] // scale, image_shape[2])

    # compile generator
    if pretrain:
        # load weights from pre-trained SRResNet
        model_path = "H:/PycharmProjects/SISR/TrainingSRCNN/RGB-ResSRCNN/RGB-ResSRCNN.h5"
        resnet_model = keras.models.load_model(model_path, custom_objects={"ssim": ssim,
                                                                           "psnr": psnr,
                                                                           "ResUnit": ResUnit})
        gen_model = keras.models.clone_model(resnet_model)
        gen_model.set_weights(resnet_model.get_weights())
    else:
        gen_model = resnetsr(input_shape, filter=64, kernel_sz=3, strides=1, no_resnet=16, upscaling=2)

    gen_model.compile(optimizer=optimizer, loss=loss.vgg_loss)

    # compile discriminator
    disc_model = disciminator(image_shape, bn_flag=True)
    disc_model.compile(optimizer=optimizer, loss="categorical_crossentropy")

    # gan

    gan = gan_model(disc_model, gen_model, input_shape, optimizer, loss.vgg_loss)

    # for epoch in range(n_epochs):

    steps_per_epoch = train_ds_size // batch_size + 1 if train_ds_size % batch_size != 0 else train_ds_size // batch_size

    for e in tqdm(range(n_epochs)):
        print("-- Epoch :%s \n" % e)
        for n, train_batch in zip(tqdm(range(steps_per_epoch)), dataset):
            # get train batch data
            image_batch_lr, image_batch_hr = train_batch[0], train_batch[1]  # train_batch is a tuple

            # phase1: train the discriminator
            generated_images = gen_model.predict(image_batch_lr)
            image_fake_and_real = tf.concat([generated_images, image_batch_hr], axis=0)
            label_fake_and_real = np.asarray([0.] * batch_size + [1.0] * batch_size)

            disc_model.trainable = True

            discriminator_loss = disc_model.train_on_batch(image_fake_and_real, label_fake_and_real)

            # phase 2: train the gan
            disc_model.trainable = False
            gan_Y = np.ones(batch_size)
            gan_loss_metric = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss : %.4f" % gan_loss_metric[0], "gan_ssim: %.4f" % gan_loss_metric[-2],
              "gan_acc: %.4f" % gan_loss_metric[-1])

        # write loss to .txt
        with open(savepath + "loss.txt", "a") as f:
            f.write("epoch %d : gan_loss = %s; discriminator_loss = %f\n" % (e, gan_loss_metric, discriminator_loss))

    gen_model.save(savepath + "gen_model.h5")
    disc_model.save(savepath + "disc_model.h5")


def train_srmodels(model, train_ds, train_ds_size, val_ds, val_ds_size, epochs, batch_size, callbacks, model_name,
                   savepath):
    """training super-resolution CNNs"""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=[ssim, psnr])
    model.summary()

    train_steps = train_ds_size // batch_size + 1 if train_ds_size % batch_size != 0 else train_ds_size // batch_size
    val_steps = val_ds_size // batch_size + 1 if val_ds_size % batch_size != 0 else val_ds_size // batch_size
    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=val_ds,
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps,
                        callbacks=callbacks)

    # ------ save results ----------
    model.save(savepath + "/" + model_name + ".h5")

    training_dataframe = pd.DataFrame(history.history)
    training_dataframe.to_pickle(savepath + "/" + model_name + ".pkl")
    fig = training_dataframe.plot().get_figure()
    fig.savefig(savepath + "/" + model_name + ".jpg")


def train_sr(model_name, data_dir, batch_size, scale, epoch, lr, save_dir, unet=False, blind_model=False, use_pca=False):
    """training photoacoustic image super-resolution CNNs"""

    # compile model
    if model_name == "resnetsr":
        model = resnetsr(filter=64, kernel_sz=3, strides=1, no_resnet=16, upscaling=scale)
    elif model_name == "densenetsr":
        model = densenetsr(filters=64, kernel_size=3, strides=1, activation="relu", blocks=4, upscaling=scale,
                           attention='csa')
    elif model_name == "fsrcnn":
        model = fsrcnn(56, 12, 4, scale)
    elif model_name == "fdunet":
        model = getModel(input_shape=(None, None, 1), filters=64, kernel_size=3, padding='same',
                         activation='relu', kernel_initializer='he_normal')
    elif model_name == "resunet":
        model = Res_UNet.getModel(input_shape=(None, None, 1), filters=32, kernel_size=3, padding='same',
                                  activation='relu', kernel_initializer='he_normal')
    elif model_name == "rdn":
        rdn = RDN(arch_params={'C': 6, 'D': 4, 'G': 32, 'G0': 64, 'x': scale},
                  patch_size=None,
                  c_dim=1,
                  kernel_size=3,
                  upscaling='shuffle',
                  init_extreme_val=0.05,
                  weights_path='')
        model = rdn._build_rdn()
    else:
        logging.error('No such model!')

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=[psnr, ssim])
    model.summary()

    # train
    data = load_train_data(data_dir)
    data = data[:, :, :, np.newaxis]
    data = data.astype('float32') / 255.0
    print(data.shape)

    psnrs = []
    ssims = []
    lr = keras.callbacks.ReduceLROnPlateau(monitor='ssim', factor=0.5)

    # load pca
    if use_pca:
        pca = generate_pca(ksize=7, dim_pca=0.95, num_kernels=1000)
    else:
        pca = np.zeros((7, 7))

    hist = model.fit(train_datagen(data, pca, down_scale=scale, batch_size=batch_size,
                                   unet=unet, blind=blind_model),
                     steps_per_epoch=len(data) // batch_size, epochs=epoch, verbose=1,
                     callbacks=[lr])
    psnrs.append(hist.history['psnr'])
    ssims.append(hist.history['ssim'])

    # save model
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else print("Saving model...")
    np.savetxt(save_dir + "/" + model_name + "_psnrs.txt", np.asarray(psnrs), fmt='%2.4f')
    np.savetxt(save_dir + "/" + model_name + "_ssims.txt", np.asarray(ssims), fmt='%1.4f')
    model.save_weights(save_dir + "/" + model_name + "_weights.h5")
    model.save(save_dir + "/" + model_name + ".h5")
    return model


if __name__ == '__main__':
    # folder = "E:\\DeepLearningMicroscopy\\DeepLearningOA\\FSRCNN\\Train\\DIV2K_valid_HR"
    # savepath = "H:\\PycharmProjects\\SISR\\train_ds\\val_DIV2K_py.h5"
    # generate_train(folder, savepath, 25, 100, 4, 100)
    # with h5py.File(savepath, "r") as f:
    #     data = f['data'][()]
    #     label = f['label'][()]
    #     print(data.shape, label.shape)
    #     fig, axs = plt.subplots(1, 2)
    #     axs[0].imshow(data[25, ::])
    #     axs[1].imshow(label[25, ::])
    #     plt.show()

    # data_dir = os.path.join(os.getcwd(), 'data/bsd500_patch100.npy')
    # data = load_train_data(data_dir)
    # data = data.astype('float32') / 255.0
    # data = data[:, :, :, np.newaxis]
    # train_gen = train_datagen(data, down_scale=2, batch_size=32, unet=True)
    # x, y = next(train_gen)

    # generate pca
    p = generate_pca(ksize=15, dim_pca=.95, num_kernels=1000)
    if p.any():
        print(p.shape)
    else:
        print('...')

    # vector = np.dot(p, kernel.flatten())

    # print(vector.shape)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(np.reshape(p.components_[0], (15, 15)), cmap='gray')
    # axs[1].imshow(np.reshape(p.components_[1], (15, 15)), cmap='gray')
    # axs[2].imshow(kernel, cmap='gray')
    # plt.show()