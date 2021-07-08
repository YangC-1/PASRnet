"""
Predict using the trained SRCNNs.
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as SSIM
from skimage.color import rgb2ycbcr
import h5py
from skimage.transform import resize
import datetime
from models.srnns import ResUnit, SE_layer, blended_attention_unit, DenseUnit
from utils.utils import ssim, psnr
tf.keras.backend.set_floatx("float32")

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ---------- custum metrics --------------
# def ssim(y_true, y_pred):
#     return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
#
#
# def psnr(y_true, y_pred):
#     # assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
#     #                                      " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
#     #                                                                                 str(y_pred.shape))
#     # snr = -10 * tf.math.log(tf.reduce_mean(tf.square(y_pred - y_true))) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
#     snr = tf.image.psnr(y_true, y_pred, max_val=1.0)
#     return snr


def my_activation(inputs):
    return tf.clip_by_value(inputs, clip_value_min=0.0, clip_value_max=1.0)


# ----- load model -----
cur_dir = os.getcwd().replace("\\", "/") + "/"
model_path = os.path.join(cur_dir, "TrainingSRCNN/RGB-ResSRCNN/RGB-ResSRCNN.h5")
model = keras.models.load_model(model_path, custom_objects={"ssim": ssim,
                                                            "psnr": psnr,
                                                            "ResUnit": ResUnit})
                                                            # "SE_layer": SE_layer})

#  ------ evaluate ------
# with h5py.File(cur_dir + "train_ds/G100_s4.h5", mode="r") as f:
#     print(list(f.keys()))
#     test_imageds = f["data"][()]
#     test_imagesz = test_imageds.shape
#     test_imageds = test_imageds.reshape((test_imagesz[0], test_imagesz[2], test_imagesz[3], test_imagesz[1]))
#     test_labelds = f["label"][()]
#     test_labelsz = test_labelds.shape
#     test_labelds = test_labelds.reshape((test_labelsz[0], test_labelsz[2], test_labelsz[3], test_labelsz[1]))
#
# test_ds = tf.data.Dataset.from_tensor_slices((test_imageds, test_labelds))
# test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
# print("buffer_size: ", test_ds_size)
# test_ds = test_ds.cache().shuffle(buffer_size=test_ds_size).batch(batch_size=16).prefetch(buffer_size=1)
#
# start_time = datetime.datetime.now()
# test_results = model.evaluate(test_ds)
# end_time = datetime.datetime.now()
# print("Time:", end_time-start_time)
# print("Evaluation results:", test_results)

# ----- Show training curve -----
# train_data = pd.read_pickle(cur_dir + "TrainingSRCNN/DenseNet/DenseNet.pkl")
# train_data["loss"].plot(xlabel="epochs", ylabel="SSIM", kind="line", figsize=(4, 3))
# train_data["val_loss"].plot(xlabel="epochs", ylabel="SSIM", kind="line", figsize=(4, 3))
# plt.legend(["loss", "val_loss"])
# plt.subplots_adjust(left=0.180, bottom=0.170)

# ---- load and predict images -----
img_path = cur_dir + "test"
scaling = 4
l_psnr, h_psnr, l_ssim, h_ssim = [], [], [], []
for root, dirs, files in os.walk(img_path, topdown=False):
    for file in files:
        # label_path = os.path.join(img_path, "file")
        label_path = img_path + "/" + file
        # load image and crop
        # label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255.0
        label_img = Image.open(label_path)
        w, h = label_img.size
        lw, lh = (w // scaling, h // scaling)
        hw, hh = (lw * scaling, lh * scaling)

        if len(np.asarray(label_img).shape) == 3:
            label_img = label_img.convert("YCbCr")
            label_img = np.asarray(label_img)
            label_img = label_img[0:hh, 0:hw, :]  # crop image
        else:
            label_img = np.asarray(label_img)[0:hh, 0:hw, :]

        label_img = Image.fromarray(label_img)

        # lr_img = cv2.resize(label_img, lr_dim, interpolation=cv2.INTER_CUBIC)
        # bicubic_img = cv2.resize(lr_img, hr_dim, interpolation=cv2.INTER_CUBIC)

        bicubic_img = np.asarray(label_img.resize((lw, lh), Image.BICUBIC).resize((hw, hh), Image.BICUBIC)) / 255.0

        lr_img = np.asarray(label_img.resize((lw, lh), Image.BICUBIC)) / 255.0
        label_img = np.asarray(label_img) / 255.0
        input_img = np.reshape(lr_img, (1, lh, lw, 3))

        start_time = datetime.datetime.now()
        predict_img = model.predict(input_img)
        end_time = datetime.datetime.now()
        print("Inference time: ", end_time-start_time)
        predict_img = np.reshape(predict_img, (hh, hw, 3))

        # save image
        if file == "kidney_2.jpg":
            lrimgDir = img_path + "/lr" + file
            bicimgDir = img_path + "/bic" + file
            hrimgDir = img_path + "/hr" + file
            gtimgDir = img_path + "/gt" + file
            cv2.imwrite(lrimgDir, lr_img * 255)
            cv2.imwrite(bicimgDir, bicubic_img * 255)
            cv2.imwrite(hrimgDir, predict_img * 255)
            cv2.imwrite(gtimgDir, label_img * 255)

        #  cal_psnr and ssim
        # b_psnr, b_ssim = psnr(label_img, bicubic_img).numpy(), SSIM(label_img, bicubic_img)
        # p_psnr, p_ssim = psnr(label_img, predict_img).numpy(), SSIM(label_img, predict_img)
        # print("Bicubic PSNR and SSIM:", b_psnr, b_ssim)
        # print("Model PSNR and SSIM:", p_psnr, p_ssim)
        # l_psnr.append(b_psnr)
        # l_ssim.append(b_ssim)
        # h_psnr.append(p_psnr)
        # h_ssim.append(p_ssim)

        #  display results
        plt.figure(figsize=(3, 3))
        plt.imshow(lr_img, cmap="gray")
        plt.title("Input")

        plt.figure(figsize=(3, 3))
        plt.imshow(bicubic_img, cmap="gray")
        plt.title("Bicubic")

        plt.figure(figsize=(3, 3))
        plt.imshow(predict_img, cmap="gray")
        plt.title("Predict")

        plt.figure(figsize=(3, 3))
        plt.imshow(label_img, cmap="gray")
        plt.title("Label")

        # plt.figure(figsize=(3, 3))
        # plt.plot(bicubic_img[280, :], label="Input")
        # plt.plot(predict_img[280, :], label="Predict")
        # plt.plot(label_img[280, :], label="Label")
        # plt.xlabel("Pixels")
        # plt.ylabel("Amplitude (a.u)")
        # plt.subplots_adjust(left=0.175, bottom=0.125)
        # plt.legend()
        plt.show()


#  save data to using DataFrame
# print(np.mean(np.asarray(l_psnr)), np.mean(np.asarray(h_psnr)), np.mean(np.asarray(l_ssim)), np.mean(np.asarray(h_ssim)))
# results = {"l_psnr": l_psnr, "h_psnr": h_psnr, "l_ssim": l_ssim, "h_ssim": h_ssim}
# df = pd.DataFrame(data=results)
# df.to_pickle(img_path + "/metrics_data.pkl")
