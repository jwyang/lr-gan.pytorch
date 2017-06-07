## Pytorch code for Layered Recursive Generative Adversarial Networks

### Introduction

This is the pytorch implementation of our ICLR 2017 paper "LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation". In our paper, we proposed a model to generate images layer-by-layer recursively. LR-GAN first generates a background image, and then generates foregrounds with appearance, pose and shape. Afterward, the foregrounds are placed on somewhere of background according to their pose and shape. By this way, LR-GAN can significantly reduce the blending between background and foregrounds. Both the qualitative and quantitative comparisons indicate that LR-GAN could generate better and sharp images than the baseline DCGAN model.

### Disclaimer

This is the reproduction code of LR-GAN based on Pytorch. Our original code was implemented based on Torch during the first author's internship. All the results presented in our paper were obtained based on the Torch code, which cannot be released since the firm restriction.

### Citation

If you find this code useful, please cite the following paper:

    @article{yang2017lr,
        title={LR-GAN: Layered recursive generative adversarial networks for image generation},
        author={Yang, Jianwei and Kannan, Anitha and Batra, Dhruv and Parikh, Devi},
        journal={ICLR},
        year={2017}
    }

### Dependencies

1. PyTorch. Install [PyTorch](http://pytorch.org/) with proper commands. Make sure you also install *torchvision*.

2. Spatial transformer network with mask (STNM). We have provided this module in this project. But if you want to do some your own changes, please refer to this [project](https://github.com/jwyang/stnm.pytorch).

### Train LR-GAN

Pull this project to your own machine, and then make sure Pytorch is installed successfully. Then, you can try to train the LR-GAN model on the following datasets:

1. CIFAR-10. CIFAR-10 is a 32x32 image dataset. We use two timesteps for the generation. The command for training is:
```bash
$ python train.py --dataset cifar10 --dataroot datasets/cifar-10 --ntimestep 2 --imageSize 32
```
