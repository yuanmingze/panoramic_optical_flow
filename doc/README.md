
# 1. Folder structure

```
.
├── paper_draft: folder is paper's draft, latex file;
├── paper_draft/image: images for documents;
├── optical_flow_evaluation.md: the document of how to evaluate performance of optical flow.
├── README.md
```

# 2. Methods 

There are two steps:

**Method 1**: Warp based method

**Method 2**: CNN based method
- Unsupervised Training, with warped image MISS:
  - Learning Stereo from Single Images
  - Learning optical flow from still images
- Change Choen's FFT to DCT, to solve the rotation invariance
- Kernel transform NN, 
- GNN the get the alignment coefficients;
- Fine Tune perspective CNN on 360 images, use panoramic image train the perspective image CNN;

## 2.1. Reference

Reference: https://medium.com/@deviparikh/planning-paper-writing-553f497e8839

# 3. Code & Data Convention

## 3.1. Data Numpy shape

The image and flow data store in Numpy array.
Image array shape is [height, width, 3], and optical flow array shape is [height, width, 2].

## 3.2. Coordinate System Convention

In this project the default coordinate system same as the following image.
- x,y and z are the cartesian coordinate system notation.
- φ and θ are the spherical coordinate system notation, which are the longitude and latitude respectively.
- u and v are the the image coordinate system notation, which are the column and row index respectively.

![coordinate System](./doc/images/coordinate_system.svg)

**Cartesian coordinate system (3D)**

The +X is right, +Y is down and +Z is forward (Right hand coordinate system), and the origin [0,0,0] is overlap with the spherical coordinate system.

**Gnomonic coordinate system (Tangent Image)**

The tangent image show in the image as a pink square.
It's generated with gnomonic projection (xy) is the gnomonic plane coordinate system (normalized coordinate) whose origin is image center, +x is right, + y is up. And the tangent point is the origin of image.

And the uv are the gnomonic image coordinate system whose origin is image top-left.

About the gnomonic projection please reference https://mathworld.wolfram.com/GnomonicProjection.html

**Spherical coordinate system**

Its origin is overlap with cartesian coordinate system.
The θ axis range is [-π, +π) which is consistent with xz plane.
And φ axis range is [-0.5 * π, + 0.5 * π) which is consistent with yz plane.

This project also use (longitude, latitude) spherical coordinate system notationm which is same as (theta, phi)

This project use a special convention introduces in Jump.
https://developers.google.com/vr/jump/rendering-ods-content.pdf

**Equirectangular Image**

The ERP image pixels coordinate origin is Top-Left, and the image row and column index are v and u, is in range [0, width) and [0, hight) respectively.

And the ERP image's spherical coordinate origin at image center, phi (latitude) and theta(theta) is (0,0).
The first pixel 0 is corresponding azimuth -π, and the last pixel image_width - 1 is corresponding +π.
The theta is [-π, +π), phi is [-0.5*π, +0.5*π].

## 3.3. Optical flow

The optical flow U is corresponding theta, and V is corresponding theta.
The optical flow data structure layers order are U and V.

**Wrap around**

The default, the ERP optical flow wrap around. 
For example, if a pixel from the [0.8𝜋,0] cross the theta [+𝜋,-] to [-0.9𝜋,0], the pixel's optical flow is +0.3𝜋 not 1.7𝜋.
The project name this optical flow as ERP optical flow.
And another type of optical flow as Non-ERP optical flow.
