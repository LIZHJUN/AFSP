# AFSP
Code for our paper "[ Image Inpainting] 

### Introduction

### Requirements

1. Pytorch >= 1.0
2. Python 3
3. NVIDIA GPU + CUDA 11.0
4. Tensorboard
5. Matlab 2018a

### Installation

1. Clone this repository

   ```bash
   git clone https://github.com/LIZHJUN/AFSP
   ```

2. Build Gaussian Sampling CUDA package 

   ```bash
   cd ./AFSP/resample2d_package
   python setup.py install --user
   ```


### Running

**1.   Image Prepare**

We train our model on three public datasets including Places2, Celeba, and Paris StreetView. We use the irregular mask dataset provided by [PConv](https://arxiv.org/abs/1804.07723). You can download these datasets from their project website.

1. [Places2](http://places2.csail.mit.edu)
2. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
3. [Paris Street-View](https://github.com/pathak22/context-encoder) 
4. [Irregular Masks](http://masc.cs.gmu.edu/wiki/partialconv)

After downloading the datasets, The edge-preserved smooth images can be obtained by using [RTV smooth method](http://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/). Run generation function [`scripts/matlab/generate_structre_images.m`](scripts/matlab/generate_structure_images.m) in your matlab. For example, if you want to generate smooth images for Places2, you can run the following code:

```matlab
generate_structure_images("path to Places2 dataset root", "path to output folder");
```

Finally, you can generate the image list using script  [`scripts/flist.py`](scripts/flist.py) for training and testing.

**2.   Training**

To train our model, modify the model config file [model_config.yaml](model_config.yaml). You may need to change the path of dataset or the parameters of the networks etc. Then run the following code:

```bash
python train.py \
--name=[the name of your experiment] \
--path=[path save the results] 
```

**3.   Testing**

To output the generated results of the inputs, you can use the [test.py](test.py).  Please run the following code:

```bash
python test.py \
--name=[the name of your experiment] \
--path=[path of your experiments] \
--input=[input images] \
--mask=[mask images] \
--structure=[structure images] \
--output=[path to save the output images] \
--model=[which model to be tested]
```

**4.   Evaluate**

To evaluate the model performance over a dateset, you can use the provided script [./scripts/metrics.py](scripts/metrics.py). This script can provide the L1 loss, PSNR, SSIM and LPIPS of the results. Besides, you can use the provided script [./scripts/FID.py](scripts/FID.py). This FID can provide the Fréchet Inception Distance ([FID score](https://github.com/mseitzer/pytorch-fid)) of the results.

```bash
python ./scripts/metrics.py \
--input_path=[path to ground-truth images] \ 
--output_path=[path to model outputs] \
--fid_real_path=[path to the real images using to calculate fid]
```

**The pre-trained weights can be downloaded from [Places2](), [CelebA-HQ](), [Paris Street]().**

Download the checkpoints and save them to './path_of_your_experiments/name_of_your_experiment/checkpoints'

For example you can download the checkpoints of Places2 and save them to './results/places2/checkpoints' and run the following code:

```bash
python test.py \
--name=places \
--path=results \
--input=./example/places/1.jpg \
--mask=./example/places/1_mask.png \
--structure=./example/places/1_tsmooth.png \
--output=./result_images \
--model=3
```

### Citation

We built our code based on [StructureFlow](https://github.com/RenYurui/StructureFlow).If you find this code is helpful for your research, please cite the following paper:

```
@inproceedings{ren2019structureflow,
      author = {Ren, Yurui and Yu, Xiaoming and Zhang, Ruonan and Li, Thomas H. and Liu, Shan and Li, Ge},
      title = {StructureFlow: Image Inpainting via Structure-aware Appearance Flow},
      booktitle={IEEE International Conference on Computer Vision (ICCV)},
      year = {2019}
}
```



### Acknowledgements

We built our code based on [StructureFlow](https://github.com/RenYurui/StructureFlow). Part of the code were derived from [Edge-Connect](https://github.com/knazeri/edge-connect) and [CA](https://github.com/JiahuiYu/generative_inpainting). Please consider to cite their papers. 

