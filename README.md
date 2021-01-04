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

## Spherical Coordinate System

There is more one spherical coordinate symbol used in this project.

The following symbols are equal:
- (longitude, latitude):
- (phi, theta)
- (lambda, phi): used in gnomonic projection;
- (azimuth, elevation)
And the corresponding range is [-pi, +pi] and [-pi/2, +pi/2], respectively.
The ERP image pixel's spherical coordinate is at the center of image.

The all 3D cartesian coordinate +x is right, +z is forward, +y is down.

The image pixel's origin is at the top-left.
