
# Folder structure

```
.
├── draft: latex file;
├── reference: the reference parpers;
├── image: images for documents;
└── optical_flow_evaluation.md: the document of how to evaluate performance of optical flow.
```

# Data  convention

## Numpy shape
The image and flow data store in Numpy array.
Image array shape is [height, width, 3], and optical flow array shape is [height, width, 2].

# Coordinate System Convention

In this project the default coordinate system same as the following image.
- x,y and z are the cartesian coordinate system notation.
- φ and θ are the spherical coordinate system notation, which are the longitude and latitude respectively.
- u and v are the the image coordinate system notation, which are the column and row index respectively.

![coordinate System](./images/coordinate_system.svg)

## Cartesian coordinate system (3D):
The +X is right, +Y is down and +Z is forward, and the origin [0,0,0] is overlap with the spherical coordinate system.


    Right hand coordinate system: 


## Gnomonic coordinate system (Tangent Image): 
The tangent image show in the image as a pink square.
It's generated with gnomonic projection (xy) which is normalized coordinate, and uv is corresponding image pixel coordinate.
The xy is the gnomonic plane coordinate system whose origin is image center.
And the uv are the gnomonic image coordinate system whose origin is image top-left.

About the gnomonic projection please reference https://mathworld.wolfram.com/GnomonicProjection.html

## Spherical coordinate system:
To easily map the spherical coordinate to Equirectangular coordinate, I use a special convention introduces in Jump.
https://developers.google.com/vr/jump/rendering-ods-content.pdf

The θ axis range is [-π, +π) which is consistent with xz plane.
And φ axis range is [-0.5 * π, + 0.5 * π) which is consistent with yz plane.

Meanwhile its origin is overlap with cartesian coordinate system.

There is more one spherical coordinate system notation used in this project.
The following symbols are equal:
- (longitude, latitude):
- (theta, phi)
- (lambda, phi): used in gnomonic projection;
- (azimuth, elevation)

## Equirectangular Image
The origin of Image coordinate [0,0] is Top-Left, and the image row and column index are v and u.

    ERP image Original is top_left, spherical coordinate origin as center.
    the point location in ERP image, the x coordinate is in range [0, width), y is in the ranage [0, hight).
    The first pixel 0 is corresponding azimuth -PI, and the last pixel image_width - 1 is corresponding (2PI) / image_width * (image_width -1 - 0.5* image_width). 

    the x coordinate is in range [0, width), y is in the ranage [0, hight)
wrap_around: if true, process the input points wrap around to make all point's x and y in the range [-pi,+pi], [-pi/2, +pi/2]
    The range of erp theta is [-pi, +pi), phi is [-0.5*pi, +0.5*pi].
    The origin of the ERP is in the Top-Left, and origin of the spherical at the center of ERP image.


# Optical flow


"""
The optical flow U is corresponding theta, and V is corresponding theta.
Conversion: 1) the arguments order is theta(U or X) and theta(V or Y)
            2) the optical flow layers order are U and V
"""

2) The gnomonic projection result (normalized tangent image) origin at the center, +x is right, + y is up. And the tangent point is the origin of image.
    In the spherical coordinate systehm the forward is +z, down is +y, right is +x.  The center of ERP's phi (latitude) and theta(longitude) is (0,0) 

The default, the ERP optical flow wrap around. 
For example, if a pixel from the [0.8𝜋,0] cross the longitude [+𝜋,-] to [-0.9𝜋,0], the pixel's optical flow is +0.3𝜋 not 1.7𝜋.
The project name this optical flow as ERP optical flow.
And another type of optical flow as Non-ERP optical flow.

# Azimuthal equidistant projection

https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection

https://fr.maplesoft.com/applications/view.aspx?SID=3583&view=html

https://casa.nrao.edu/aips2_docs/memos/107/node2.html

https://casa.nrao.edu/aips2_docs/memos/107/node2.html#SECTION00021100000000000000

http://www.geography.hunter.cuny.edu/~jochen/GTECH361/lectures/lecture04/concepts/Map%20coordinate%20systems/Perspective.htm
