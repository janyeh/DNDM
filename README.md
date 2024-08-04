# DNDM
The Pytorch implementation for "Semi-supervised Single-Image Dehazing Network via Disentangled Meta-Knowledge".

# Paper   
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10202595
![image](https://github.com/dehazing/DNDM/assets/100458096/458b7df5-4794-41a8-b239-f70d1a1b02f9)

# Prerequisites
Python 3.6
PyTorch 1.1.0
Ubuntu 
Anaconda

# Installation

1. Setup the Conda environment
```bash
conda create -n dndm python=3.6
```
```bash
conda activate dndm
```
```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```
```bash
pip install scikit-image natsort Pillow cmake cython opencv-python==4.1.2.30 visdom
```

2. Setup the Directories 
```bash 
cd Code/
bash setup.sh
```

# Usage 

1. Open terminal inside the `Code/` directory
```bash
cd Code/
conda activate dndm
visdom
```

2. Open a new terminal inside `Code/` directory
```bash
cd Code/
conda activate dndm
python train.py 
```


# Citation
@ARTICLE{10202595,
  author={Jia, Tongyao and Li, Jiafeng and Zhuo, Li and Yu, Tianjian},
  journal={IEEE Transactions on Multimedia}, 
  title={Semi-supervised Single-Image Dehazing Network via Disentangled Meta-Knowledge}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2023.3301273}}
