## DCGAN
![](https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/test_2017-01-11%2009:40:47.png)
Fig. Some generated samples trained on CUB-200-2011.

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- CUDA 8.0

Add Cuda Path to bashrc first
```bash
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
```

We recommend you to install anaconda. Here we write a simple script for you to install the dependence by anaconda.
```python
# install env (especially for old version Tensorflow)
conda env create -f dcgan.yml
# activate env, then you can run code in this env without downgrading the outside Tensorflow.
source activate dcgan
```

### Let's start
We did some slight changes compare to the original code. You can download the [my forked code](https://github.com/layumi/DCGAN-tensorflow) first and then modify the codes as we did.

### 1.Train

`python main.py --dataset duke_train --train --input_height 128 --output_height 128 --options 1`

`duke_train` is the dir path which contains images. Here I use the (DukeMTMC-reID)[https://github.com/layumi/DukeMTMC-reID_evaluation] training set. You can change it to your dataset path.

### 2.Test

`python main.py --dataset duke_train --options 5`  
It will use your trained model and generate 48,000 images for the following semi-supervised training.
