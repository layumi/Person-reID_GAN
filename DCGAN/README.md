## DCGAN

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)

Add Cuda Path to bashrc first
```bash
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
```

We write a simple script for you to install the dependence.
```python
# install env (especially for old version Tensorflow)
conda install -f dcgan.yml
# activate env, then you can run code in this env without downgrade the Tensorflow outside.
source activate dcgan
```

### Let's start
We did some slight changes compare to the original code. You can download the [original code](https://github.com/carpedm20/DCGAN-tensorflow) first and then modify the codes as we did.


![](https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/test_2017-01-11%2009:40:47.png)
Fig. Some generated samples trained on CUB-200-2011.
### 1.Deepen the network.

The original network is trained on 64x64 input images. We slightly change the generator and discriminator network strucuture, which can recieve 128x128 input images. 

Generator(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/model.py#L293)

Discriminator(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/model.py#L253)

### 2.Output Size

We noticed that directly training on 256x256 input images will lead to random noisy images.

So we use DCGAN to generate 128x128 output then resize it to 256x256 for further training.
(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L48)

### 3.Random vector
We use `option=5` to generate images. You can change the range of input random vector.
(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L223)

### 4.Train

Change the visualization setting from `OPTION = 5` to ` OPTION = 1` (https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/main.py#L71)

Uncomment (https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L50)

Comment (https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L48)

Then Train: `python main.py --dataset duke_128 --train`

`duke_128` is the dir path which contains images. You can change it to your dataset path.

### 5.Test
Change the visualization setting from `OPTION = 1` to ` OPTION = 5` (https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/main.py#L71)

Uncomment (https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L48)

Comment (https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L50)

Test: `python main.py --dataset duke_128`  

It will use your trained model and generate 48,000 images for the following semi-supervised training.
