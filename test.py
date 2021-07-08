import argparse
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import logging
from tensorflow import keras
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from models.srnns import *
from utils import *
import time
from models import Res_UNet, rdn
tf.keras.backend.set_floatx("float32")

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='resunet', type=str, help='name of the saved model')
parser.add_argument('--model_path', default='./PASR_results/resunet_scale4/resunet_weights.h5', type=str, help='path of the saved model')
parser.add_argument('--load_weights', default=1, type=int, help='load saved model weights, 1 or 0')
parser.add_argument('--set_dir', default='E:/DeapLearningMedImage/05_Datasets/Duke_ORPAM/clean_valid', type=str, help='directory of test dataset')
parser.add_argument('--sigma', default=1.6, type=float, help='gaussian kernel sigma')
parser.add_argument('--scale', default=4, type=int, help='downsampling factor')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--unet', default=True, type=bool, help='Unet model')
parser.add_argument('--pca', default=False, type=bool, help='use pca')
parser.add_argument('--result_dir', default='./test/resunet_scale4', type=str, help='directory of results')
parser.add_argument('--save_result', default=1, type=int, help='save the upsampled image, 1 or 0')
parser.add_argument('--display_result', default=0, type=int, help='display results on fly, 1 or 0')
args = parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line: %(lineno)d] - %(levelname)s: %(message)s')
    assert os.path.exists(args.model_path) == 1, "Model path does not exist!"

    if args.load_weights:
        # compile model
        if args.model_name == "resnetsr":
            model = resnetsr(filter=64, kernel_sz=3, strides=1, no_resnet=16, upscaling=args.scale)
        elif args.model_name == "densenetsr":
            model = densenetsr(filters=64, kernel_size=3, strides=1, activation="relu", blocks=4, upscaling=args.scale,
                               attention='csa')
        elif args.model_name == "fsrcnn":
            model = fsrcnn(56, 12, 4, args.scale)
        elif args.model_name == "fdunet":
            model = getModel(input_shape=(None, None, 1), filters=64, kernel_size=3, padding='same',
                             activation='relu', kernel_initializer='glorot_normal')
        elif args.model_name == "resunet":
            model = Res_UNet.getModel(input_shape=(None, None, 1), filters=32, kernel_size=3, padding='same',
                                      activation='relu', kernel_initializer='he_normal')
        elif args.model_name == "rdn":
            rdn = RDN(arch_params={'C': 6, 'D': 4, 'G': 32, 'G0': 64, 'x': args.scale},
                      patch_size=None,
                      c_dim=1,
                      kernel_size=3,
                      upscaling='shuffle',
                      init_extreme_val=0.05,
                      weights_path='')
            model = rdn._build_rdn()
        else:
            logging.error('No such model!')
        # # model = resnetsr(filter=64, kernel_sz=3, strides=1, no_resnet=16, upscaling=args.scale)
        # # model = fsrcnn(56, 12, 4, args.scale)
        # rdn = RDN(arch_params={'C': 6, 'D': 4, 'G': 32, 'G0': 64, 'x': args.scale},
        #           patch_size=None,
        #           c_dim=1,
        #           kernel_size=3,
        #           upscaling='shuffle',
        #           init_extreme_val=0.05,
        #           weights_path='')
        # model = rdn._build_rdn()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss="mse", metrics=[psnr, ssim])
        # model(tf.zeros((1, 128, 128, 1)))
        model.load_weights(args.model_path)
    else:
        model = load_model(args.model_path, custom_objects={'psnr':psnr,
                                                            'ssim':ssim})
    logging.info('Model/weights has been loaded.')

    psnr, ssim, ref_time = [], [], []

    if args.pca:
        pca = generate_pca(ksize=7, dim_pca=0.95, num_kernels=1000)
        gauss_kernel = cv2.getGaussianKernel(ksize=7, sigma=args.sigma)
        gauss_kernel = gauss_kernel * np.transpose(gauss_kernel, (1, 0))
        degradation_vector = np.dot(pca, gauss_kernel.flatten())

    for im in os.listdir(args.set_dir):
        if im[-4:] in ['.jpg', '.png', '.tif', '.bmp'] or im[-5:] in ['.jpeg']:

            # load clean image and generate noisy image
            x = np.array(cv2.imread(os.path.join(args.set_dir, im), flags=cv2.IMREAD_GRAYSCALE))  # convert grayscale image
            # x = cv2.resize(x, (1024, 512), interpolation=cv2.INTER_CUBIC)
            x = x.astype('float32') / 255.0

            # if args.unet:
            h, w = int(2**np.floor(np.log2(x.shape[0]))), int(2**np.floor(np.log2(x.shape[1])))
            x = x[0:h, 0:w]
            # else:
            #     x = x[0:x.shape[0] - x.shape[0] % args.scale, 0:x.shape[1] - x.shape[1] % args.scale]

            np.random.seed(seed=0)

            blur_x = cv2.GaussianBlur(x, ksize=(7, 7), sigmaX=args.sigma, sigmaY=args.sigma, borderType=cv2.BORDER_REPLICATE)
            print('Image shape: ', blur_x.shape)

            if args.unet:
                mask_x = np.asarray(
                    list(map(lambda x: 1 if x % args.scale == 0 else 0, list(range(blur_x.shape[0])))))
                mask_y = np.asarray(
                    list(map(lambda x: 1 if x % args.scale == 0 else 0, list(range(blur_x.shape[1])))))
                downsample_x = blur_x * mask_x[:, np.newaxis] * mask_y[np.newaxis, :]
            else:
                downsample_x = blur_x[::args.scale, ::args.scale]

            if args.pca:
                degradation_map = degradation_vector[np.newaxis, np.newaxis, :] * np.ones((downsample_x.shape[0], downsample_x.shape[1], 1))
                downsample_x = np.concatenate((downsample_x[:, :, np.newaxis], degradation_map), axis=-1)

            y_ = to_tensor(downsample_x)

            # inference of the model
            start_time = time.time()
            x_ = model.predict(y_)
            elasped_time = time.time() - start_time
            ref_time.append(elasped_time)
            print('Model inference time: %s: %2.4f' % (im, elasped_time))

            # compare psnr and ssim
            x_ = from_tensor(x_)
            psnr_x_ = compare_psnr(x, x_, data_range=1.0)
            ssim_x_ = compare_ssim(x, x_, data_range=1.0)

            # Bicubic
            # x_ = cv2.resize(downsample_x, x_.shape.__reversed__(), interpolation=cv2.INTER_CUBIC)

            # show results
            if args.display_result:
                fig, axes = plt.subplots(1, 3)
                axes[0].imshow(x, cmap='gray')
                axes[1].imshow(downsample_x, cmap='gray')
                axes[2].imshow(x_, cmap='gray')
                plt.show()

            if args.save_result:
                im_name, ext = os.path.splitext(im)
                os.makedirs(args.result_dir) if not os.path.exists(args.result_dir) else print('Saving results...')
                save_result(x, path=os.path.join(args.result_dir, 'gt' + im_name + ext))
                save_result(downsample_x, path=os.path.join(args.result_dir, 'input' + im_name + ext))

            psnr.append(psnr_x_)
            ssim.append(ssim_x_)

    psnr_avg = np.mean(psnr)
    ssim_avg = np.mean(ssim)
    time_avg = np.mean(ref_time)
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    ref_time.append(time_avg)

    if args.save_result:
        results = np.hstack([psnr, ssim, ref_time])
        save_result(results, path=os.path.join(args.result_dir, 'metrics_results.txt'))

    logging.info('Data: {0:10s} \n PSNR = {1:2.4f}\n SSIM = {2: 1.4f}\n Ave_time = {3:2.4f}'
                 .format(args.set_dir, psnr_avg, ssim_avg, time_avg))
