import argparse
from utils import *
import tensorflow as tf

tf.keras.backend.set_floatx("float32")

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# load params
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='fdunet', type=str, help='the name of the model')
parser.add_argument('--data_dir', default='./data/DukePAM_patch128.npy', type=str, help='path of train data')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--scale', default=2, type=int, help='down scaling factor')
parser.add_argument('--epoch', default=50, type=int, help='number of train epochs')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_dir', default='./PASR_results/fdunet_x2', type=str, help='path for saving results')
parser.add_argument('--unet', default=True, type=bool, help='unet or not')
parser.add_argument('--blind_model', default=False, type=bool, help='blind model')
parser.add_argument('--pca', default=False, type=bool, help='use PCA for kernel reduction')
args = parser.parse_args()


if __name__ == "__main__":
    train_sr(args.model_name, args.data_dir, args.batch_size, args.scale, args.epoch, args.lr, args.save_dir,
             args.unet, args.blind_model, args.pca)


