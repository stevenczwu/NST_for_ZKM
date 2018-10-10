### NST_for_ZKM
Neural Style Transfer models trained for ZKM project


### Run manually
The project has been set to run automatically when system is rebooted for ZKM. 

In case of preferring to run the project manually, remove 'zkm project' in the 'Startup Applications Preferences'. And then, to run it manually:
```bash
cd ~/zkm
bash zkm.sh
```
( Reboot may be needed before running the commands. )


### Setup
With a new computer installed with linux system Ubantu 16.04 and GTX 1080, you may set the computer up for the project by doing following steps:

1) Install Nvidia Driver for GTX 1080

One possible method:
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
sudo apt-get install mesa-common-dev
sudo apt-get install freeglut3-dev
```
Reboot to make it take effect.

2) Install [Torch](http://torch.ch/)

First [install Torch](http://torch.ch/docs/getting-started.html#installing-torch), then
update / install the following packages:
```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
```

Install packages needed for processing the videostream data from the camera by running:
```bash
luarocks install camera
luarocks install qtlua
```

cd to your Torch folder, update Torch by running:
```bash
sudo bash update.sh
```
( Without this update, Torch may met some problem with models trained with instance normalization. )

3) Install CUDA

First [install CUDA](https://developer.nvidia.com/cuda-downloads), recommended version: cuda 8.0

!!!IMPORTANT!!!

When installing cuda, DO NOT install new graphics driver like this:

'''
......

Do you accept the previously read EULA?

accept/decline/quit: accept

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 361.62?

(y)es/(n)o/(q)uit: n

Install the CUDA 8.0 Toolkit?

(y)es/(n)o/(q)uit: y

......

'''

Add environment variables:
```bash
vim ~/.bashrc
#add lines at the bottom of the file:
    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
source ~/.bashrc
```

Update / install the following packages:
```bash
luarocks install cutorch
luarocks install cunn
```

4) Install cuDNN

First [download cuDNN](https://developer.nvidia.com/cudnn), recommended version: 5.0

Copy the following files into the CUDA Toolkit directory, and change the file permissions:
```bash
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

Environment variables have been added in cuda's installation step, no new PATH need to be added.

Then install the Torch bindings for cuDNN:
```bash
luarocks install cudnn
```

5) Clone the git and run

```bash
https://github.com/stevenczwu/NST_for_ZKM.git
rm NST_for_ZKM zkm
cd zkm
bash zkm.sh
```

Or if you want it to run automatically when the computer rebooted, change it into excutable in property-permissions of file zkm.sh, and add it to the startup applications as well.

### Change the models
I have prepared about 20 trained models for you. You may select whichever you would like to use by editing file ~/zkm/zkm.sh. Models may be added/removed after '-models', style images may be added/removed after '-style_images'.

Remember always give the corresponding style images. The number of models and style images must match, otherwise errors may occured.

Or you can run it with the default settings (3 models loaded) like:
```bash
cd ~/zkm
qlua zkm.lua
```

