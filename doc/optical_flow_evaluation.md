
# Comparison

## Method
Optical Flow Methods list:
- DIS [3];
- PWC-Net[];
- FlowNetx[];
- Sun IJCV[];

## Result

The 
Others optical flow list:
- DIS


# Optical flow GT Generation

## Convention
To define the optical flow warp around function.
There are two methods to express the optical flow:
- w/o warp around: ignore the ERP camera model, make the 360 image as normal images;
- warp around: make the pixel warp around.

## Panoramic Image Coordinate system

There is more detail about the corresponding coordinate system between the panoramic image can be found in [1].


# Dataset

Synthetic Datasets:
- Replica360;


## Replica_360 Optical Flow Ground Truth

This data is rendered by replica 360 OpenGL code.

Scene List:s
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




# Reference

1. LZMA c++ SDK: https://www.7-zip.org/sdk.html
2. LZMA python: https://docs.python.org/3/library/lzma.html
3. DIS Optical Flow: https://docs.opencv.org/3.4.11/da/d06/classcv_1_1optflow_1_1DISOpticalFlow.html
4. Rendering-ods-content.pdf: https://developers.google.com/vr/jump/rendering-ods-content.pdf
