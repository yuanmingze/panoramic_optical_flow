# File Structure

The project file structure are like following image:

```
.
├── code
|   ├── README.md
│   └── python                         # the python used to load & visual & Evaluation
│
├── data
|   ├── README.md
│   └── replica_360_hotel_0 
|
├── doc
|   ├── README.md
|   ├── image                            # the images used in the documents
│   └── optical_flow_ground_truth.md   
|
└── README.md
```

Please read the README.md file in each folder to get more information.

# Convention 

##  coordinate system

Gnomonic coordinate system: The +Y is up the +X is right, and the origin [0,0] is image center.

The ERP image spherical coordinate system: 
The +Y is up and +X is right, and the origin [0,0] is image center.

TODO theta +0.5pi is on the top?
The X axis range is [-pi, +pi) and Y axis range is [-0.5 * pi, + 0.5 * pi).


Gnomonic/ERP image coordinate system: The +Y is down ans +X is right, and the origin [0,0] is Top-Left.

3D Cartesian coordinate system: The +X is right, +Z is forward and +Y is down.


## Spherical Coordinate System naming

There is more one spherical coordinate system notation used in this project.

The following symbols are equal:
- (longitude, latitude):
- (phi, theta)
- (lambda, phi): used in gnomonic projection;
- (azimuth, elevation)


## Optical flow

The default, the ERP optical flow wrap around. 
For example, if a pixel from the [0.8𝜋,0] cross the longitude [+𝜋,-] to [-0.9𝜋,0], the pixel's optical flow is +0.3𝜋 not 1.7𝜋.
The project name this optical flow as ERP optical flow.
And another type of optical flow as Non-ERP optical flow.


## Data Order convention

The image and flow data store in Numpy array.
Image array shape is [height, width, 3], and optical flow array shape is [height, width, 2].
