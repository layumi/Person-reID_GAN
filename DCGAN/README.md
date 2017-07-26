### DCGAN

We did some slight changes compare to the original code. You can download the [original code](https://github.com/carpedm20/DCGAN-tensorflow) first and then modify the codes as we did.

1. Deepen the network which can recieve 128x128 input images. 

Generator(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/model.py#L293)

Discriminator(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/model.py#L253)

2. Output Size

We noticed that directly training on 256x256 input images will lead to random noisy images.

So we use DCGAN to generate 128x128 output then resize it to 256x256 for further training.
(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L48)

3. Random vector
We use `option=5` to generate images. You can change the range of input random vector.
(https://github.com/layumi/Person-reID_GAN/blob/master/DCGAN/utils.py#L223)

4. Train & Test

Train: `python main.py --dataset duke_128 --train`

duke_128 is a dir which contain images.

Test: `python main.py --dataset duke_128`
