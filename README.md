# PASRnet
Photoacoustic super resolution
# Towards multi-degradation photoacoustic super resolution network with channel and spatial wise attention
Photoacoustic microscopy (PAM) is a scanning-based imaging technique that is capable of resolving 
optical absorption at micrometer scale via ultrasonic detection. The image formation can be modeled 
as a blurring and down-sampling process, in which the original absorption distribution is first blurred 
by the focused light and then spatially down-sampled at scanning ratio. Deconvolution based 
algorithms can be used to correct the optical blur, but fail to restore the fully sampled image. Here, we 
propose an efficient photoacoustic super resolution network (PASRnet) that directly learns deblurring 
and up-sampling to reconstruct the original absorption distribution in PAM.
## Network Architecture
The proposed PASRnet leverages the representation ability of modern convolutional neural network 
(CNN) to perform super resolution reconstruction from the degraded image in an end-to-end manner. 
We adopt a dense connection architecture that consists of four dense blocks composed of six cascaded 
convolutional layers. We incorporated the channel and spatial wise attention module 
(CASM) to the dense blocks to guide the network to concentrate on the informative parts. To address 
multiple degradations, we stretch the blur kernels in space and input them to the network as a priori. [SRMD](https://github.com/cszn/SRMD#learning-a-single-convolutional-super-resolution-network-for-multiple-degradations).
![architecture_SRPA-Net](https://user-images.githubusercontent.com/68590273/124863666-1a35b200-dfea-11eb-9562-82e207a4c5a1.png)
## Dataset
The network is trained on the [Duke PAM dataset](https://zenodo.org/record/4042171#.YGZxzFUzaUk) which contains 372 images of mouse vasculature. 
Variation and scaling are applied on the images to generate a representative dataset. The high 
resolution images are blurred by a randomly selected blur kernel and down-sampled to 
simulate the low resolution images, which are fed to the network during training phase.
## Results
![MD_results](https://user-images.githubusercontent.com/68590273/124869961-8ddcbc80-dff4-11eb-8497-94719d6c5929.png)
