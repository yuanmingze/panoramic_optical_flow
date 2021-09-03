
# 1. Comparison

## 1.1. Method
Optical Flow Methods list:
- DIS [3];
- PWC-Net[];
- FlowNetx[];
- Sun IJCV[];

## 1.2. Host

### 1.2.1. DIS

The OpenCV DIS flow method.
The code at `code/python/utility/flow_estimate.py`

### 1.2.2. PWC-Net

Code is [official released code](https://github.com/NVlabs/PWC-Net).

1. Caffe Code (Linux):
With the PWC-Net office Caffe Code. And use the python code `python/compare_pwc-net.py` to generate the `img1.txt`, `img2.txt` and `out.txt` for the `proc_images.py` file.
- Virtual env: `/mnt/sda1/workenv_linux/python_2_7_pwcnet/`
- Caffe code `/mnt/sda1/workspace_linux/PWC-Net-flownet2`
- The `compare_pwc_net.py` generate the img1.txt, img2.txt and out.txt files.
- Run the script `source set-env.sh` than `proc_images.py` python script as `python ./proc_images.py /home/mingze/sda1/workdata/opticalflow_data_bmvc_2021/pwcnet/img1.txt /home/mingze/sda1/workdata/opticalflow_data_bmvc_2021/pwcnet/img2.txt /home/mingze/sda1/workdata/opticalflow_data_bmvc_2021/pwcnet/out.txt`
- PWC-Net model: `/mnt/sda1/workspace_linux/PWC-Net/Caffe/`

2. PyTorch Code (Slow, don't know why):
Run time environment at: Python virtual environment: `/mnt/sda1/workenv_linux/python_2_7_pytorch/`

### 1.2.3. FlowNet2

Code is [flownet2](https://github.com/lmb-freiburg/flownet2)
- Python virtual environment: `/mnt/sda1/workenv_linux/python_2_7_pytorch/`
- FlowNet2 model" `/mnt/sda1/workspace_linux/flownet2-lmb_freiburg/models/`
- FlowNet2 code: `/mnt/sda1/workspace_linux/flownet2-lmb_freiburg/`
 1. The code `code/python/compare_flownet2.py` generate the list file for `run-flownet-many.py`.
 1. `source set-env.sh`
 1. Estimate the optical flow files with 
    `python ./scripts/run-flownet-many.py ./models/FlowNet2-CSS/FlowNet2-CSS_weights.caffemodel.h5 ./models/FlowNet2-CSS/FlowNet2-CSS_deploy.prototxt.template /mnt/sda1/workdata/opticalflow_data/replica_360/apartment_0/flownet2/replica_listfile.txt`

### 1.2.4. RAFT (ECCV 2020)

Get the code from [GitHub](https://github.com/princeton-vl/RAFT)

- Windows:
Conda venv: `D:\workenv_windows\conda_raft`
Run script: `D:\source_code\RAFT\compare_raft_bmvc2021.py`

`python compare_raft_bmvc2021.py --model=models/raft-things.pth `


- Linux:
Conda install in: `/home/mingze/anaconda3` run int Ubuntu 18.04.5
 1. `conda create -p /mnt/sda1/workenv_linux/conda_raft_py3.6 python=3.6`
 2. `conda install -c menpo opencv3`

Inference:
- Conda virtual environment: `/mnt/sda1/workenv_linux/conda_raft_py3.6/`
 1. `conda activate /mnt/sda1/workenv_linux/conda_raft_py3.6/`

- Get the flo file with the RAFT code `/mnt/sda1/workspace_linux/RAFT/compare_raft.py`, which is symbol link of `code/python/compare_raft.py`.
 1. `python compare_raft.py --model=models/raft-things.pth --path=/mnt/sda1/workdata/opticalflow_data/replica_360/apartment_0/replica_seq_data/`

- RAFT model: `/mnt/sda1/workspace_linux/RAFT`


### 1.2.5. Sun IJCV

Code is [IJCV](http://cs.brown.edu/~dqsun/code/ijcv_flow_code.zip)
- Matlab code: `/mnt/sda1/workspace_windows/IJCV_2013_matlab_opticalflow/ijcv_flow_code`


### 1.2.5. OmniFlowNet

It's base on Caffe only run on Linux.
Code: `/mnt/sda1/workspace_linux/OmniFlowNet/OmniFlowNet/models/testing/`

`python ./compare_omniflownet.py`
setup venv : `/mnt/sda1/workenv_linux/python_2_7_omniflownet/bin/activate`



## 1.3. Output data

The output optical flow *.flo file, save to the original dataset folder.
The input of the scripts is the folder of `replica_seq_data` of each dataset.
And the output data folders are named with :
1. raft:
1. pwc_net:
1. flownet2:

## 1.4. Garlick

## 1.5. Evaluation Result

The test ground truth data store in the folder:

Test the optical flow methods on the fellowing datasets:
- Replica_360:
1. hotel_0: 
2. apartment_0: 
3. office_0:
4. office_4:
5. room_0: 
6. room_1: 

- 

# 2. Optical flow GT Generation

## 2.1. Convention
To define the optical flow warp around function.
There are two methods to express the optical flow:
- w/o warp around: ignore the ERP camera model, make the 360 image as normal images;
- warp around: make the pixel warp around.

## 2.2. Panoramic Image Coordinate system

There is more detail about the corresponding coordinate system between the panoramic image can be found in [1].


# 3. Dataset

Synthetic Datasets:
- Replica360;


## 3.1. Replica_360 Optical Flow Ground Truth

This data is rendered by replica 360 OpenGL code.

Scene List:
- Apartment_0:
- hotel_0: [Download](https://drive.google.com/file/d/16KheF7FRAMM3yotJXeL9V2-a46yvUbxX/view)
- office_0:
- office_4:
- room_0:
- room_1:

Folder structure:

```
.
├── %04d_opticalflow_backward.flo: optical flow file;
├── %04d_opticalflow_forward.flo: optical flow file;
├── %04d_depth.dpt: depth map file;
├── %04d_rgb.jpg: rgb image file;
├── %04d_occlusion.png: occlusion data from frame 04d;
└── info.xml: storage the basic information of the data.
```

Compressed optical flow or depth map by LZMA algorithm [1][2] named with postfix *.xz'.


# 4. Reference

1. LZMA c++ SDK: https://www.7-zip.org/sdk.html
2. LZMA python: https://docs.python.org/3/library/lzma.html
3. DIS Optical Flow: https://docs.opencv.org/3.4.11/da/d06/classcv_1_1optflow_1_1DISOpticalFlow.html
4. Rendering-ods-content.pdf: https://developers.google.com/vr/jump/rendering-ods-content.pdf
