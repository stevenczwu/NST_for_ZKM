### NST_for_ZKM
Neural Style Transfer models trained for ZKM project


### Run manually
The project has been set to run automatically when system is rebooted for ZKM. 

In case of preferring to run the project manually, remove 'zkm project' in the 'Startup Applications Preferences'. And then, to run it manually:
```bash
cd ~/zkm
bash zkm.sh
```
(Reboot may be needed before that.)


### Setup

1) Install Nvidia Driver for GTX1080

One possible method:
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
sudo apt-get install mesa-common-dev
sudo apt-get install freeglut3-dev
```
Reboot to make it take effect.

2) Install [Torch](http://torch.ch/).

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
(Without this update, Torch may met some problem with models trained with instance normalization.)

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
export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
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

Add environment variables:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
source ~/.bashrc
```

Then install the Torch bindings for cuDNN:
```bash
luarocks install cudnn
```

### Change the models
You can use the script `webcam_demo.lua` to run one or more models in real-time
off a webcam stream. To run this demo you need to use `qlua` instead of `th`:

```bash
qlua webcam_demo.lua -models models/instance_norm/candy.t7 -gpu 0
```

You can run multiple models at the same time by passing a comma-separated list
to the `-models` flag:

```bash
qlua webcam_demo.lua \
  -models models/instance_norm/candy.t7,models/instance_norm/udnie.t7 \
  -gpu 0
```
The webcam demo depends on a few extra Lua packages:
- [clementfarabet/lua---camera](https://github.com/clementfarabet/lua---camera)
- [torch/qtlua](https://github.com/torch/qtlua)

You can install / update these packages by running:


