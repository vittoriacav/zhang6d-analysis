# An In-depth Performance Analysis of 6DoF Object Pose Estimation
This repository contains the code used to develop the final thesis of the second cycle degree in Artificial Intelligence Systems at the University of Trento.

Supervisor: Nicola Conci

Co-supervisor: Lorenzo Orlandi

This work is heavily relies on the work of Zhang et al.
> Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild  
> Kaifeng Zhang, Yang Fu, Shubhankar Borse, Hong Cai, Fatih Porikli, Xiaolong Wang  
> arXiv, 2022  
> [Project Page](https://kywind.github.io/self-pose) / [ArXiv](https://arxiv.org/abs/2210.07199) / [GitHub Repo](https://github.com/kywind/self-corr-pose)

## Environment Setup
PyTorch with CUDA support are required. The code is tested on python 3.8, torch 1.10.0, CUDA 11.1, and RTX 3090.

Check ubuntu version
```sh
lsb_release -a
```
Check CUDA version (substitute with your path)
```sh
/usr/local/cuda/bin/nvcc --version
```

Install python 3.8 along another version if needed
```sh
# System update
sudo apt update && sudo apt upgrade
# Add PPA for Python old versions
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
# Check Python Versions you want
sudo apt-cache policy python3.8
# Install python 3.8
sudo apt install python3.8
# Set the default Python version
# Check what python versions are available on your systems:
ls /usr/bin/python*
# To know whether any version is configured as python alternatives or not
sudo update-alternatives --list python
# If the output is: “update-alternatives: error: no alternatives for python”
# Then it means there are no alternatives that have been configured
# Here we are setting up two versions as alternatives, later we will be able 
# to choose between them to set one of them as the system default one
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2
# Switch the default Python version and enter the Selection number of the Python 
# you want to set as the system’s default one.
sudo update-alternatives --config python
# check python version
python -V
```

Create virtualenv with Python 3.8
```sh
sudo apt install python3-virtualenv
virtualenv --python="/usr/bin/python3.8" "/YOUR/PROJECT/PATH"
source /YOUR/PROJECT/PATH/bin/activate
cd /YOUR/PROJECT/PATH
```

Clone repo: https://github.com/vittoriacav/zhang6d-analysis.git

If you are in Visual Studio Code

ctrl+shift+p

Git:Clone 

https://github.com/vittoriacav/zhang6d-analysis.git

```cd zhang6d-analysis```


When trying to install required packages if you encounter the error:
```sh
ModuleNotFoundError: No module named 'distutils.cmd'
```
Then install ```python3.8-distutils```
```sh
sudo apt install python3.8-distutils
```

Environment setup required from authors
```sh
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install absl-py tensorboard opencv-python setuptools==59.5.0 trimesh kornia fvcore iopath matplotlib wandb scikit-learn scipy
```

How to install pytorch3d that gives problems without anaconda: https://github.com/facebookresearch/pytorch3d/issues/1184 
```sh
cd ~/your/virtualenv/path
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git tag -l
git checkout v0.6.1
pip install -e .
```
Check see pytorch version
```sh
pip show pytorch3d
```
Install requirements, might give problems: ignore them
```sh
pip install -r requirements.txt
```

Install SoftRas (instructions from authors):

After git cloning the repository, build the required [SoftRas](https://github.com/ShichenLiu/SoftRas) package located in ```/YOUR/PROJECT/PATH/zhang6d-analysis/third-party/softras```:
```sh
cd third-party/softras
python setup.py install
```

If you encounter error:
```sh
error: Setup script exited with error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```
Solution from [StackOverflow](https://stackoverflow.com/questions/26053982/setup-script-exited-with-error-command-x86-64-linux-gnu-gcc-failed-with-exit)

```sh
sudo apt-get install python3.8-dev
cd /YOUR/PROJECT/PATH/zhang6d-analysis/third-party/softras
python setup.py install
```

The model uses a pretrained [DINO](https://github.com/facebookresearch/dino) for correspondence learning. Download the pretrained models from [here](https://drive.google.com/drive/folders/1MOeWKoHoBK9GH6jJ-BZPvD9rj9xQdWux?usp=share_link) and put them in the `/YOUR/PROJECT/PATH/zhang6d-analysis/pretrain/` directory. Also create the ```/YOUR/PROJECT/PATH/zhang6d-analysis/log/``` directory for logging. You can also assign custom paths via configuration.

```sh
cd /YOUR/PROJECT/PATH/zhang6d-analysis
mkdir pretrain
cd pretrain
```
Use [gdown](https://github.com/wkentaro/gdown) to download files from GoogleDrive. For example:
```sh
gdown jWTd8Uftv9y5RdzqlqQ18MhRkFFbquyc
```
