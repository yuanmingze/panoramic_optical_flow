# 1. Introduction

The code use to generate sythetic data for MageParallax.

## 1.1. Runtime env

The oirginal position of camera is (0.0, 0.0, 0.0)
The orientation of camera is (0.0, 0.0, 0.0).
Up is (0.0, 0.0, 1.0), forward is (1.0, 0.0, 0.0), left is (0.0, 1.0, 0.0).

## 1.2 scene

Use the fellowing scenes 

1. apartment_0
1. hotel_0
1. office_0
1. office_1
1. office_4
1. room_0
1. room_1
1. room_2

The fellowing scenes are not closure scene, show as the images in the folder 'images'

- apartment_1
- apartment_2
- frl_apartment_0
- frl_apartment_1
- frl_apartment_2
- frl_apartment_3
- frl_apartment_4
- frl_apartment_5
- office_2
- office_3


## 1.2. 3rd-part dependent

Plantform: Linux

Libraries:
- EGL
- OpenCV
- Eigne 3.3.7
- Pangolin 
- Python
- OpenMP
- [flow-io-opencv](https://github.com/davidstutz/flow-io-opencv.git)

Software:
- Blender 2.8.2 or later


## 1.3 Convention

- frame index start from 0;
- image extention name is `*.png`, `*.jpg`


# 2. Tutorial

The post-process will generate two kinds of data.
One use for MegaParallalx `Preprocessing.exe` input, another use for the MegaParallax `Viewer.exe` input.

## 2.1. step 1: generate the camera trajectory file for Replica Rending

Use the file `megeparallax/gen_video_path_mp.py` to generate a `*.csv` file.
Before run the python script, change the `scene_name` and `nsteps`, e.t.c. value in the python.

- `scene_name`: the scene name is cooresponding the file `glob\$scene_name$.txt`;
- `nsteps`: how many frames will generate in the full circle;
- `radius`: the radius of the circle with unit is meter;
- `output_dir`: the trajectory file output folder;
-  The otuput trajectory file name with `datasetname_YYYY_MM_DD_HH_mm_[circle/square/line_path][_center].csv`.

The script will create to `*.csv` files.
One is the whole camera trajectory, another is the center position of the scene.

The camera coordinate system is as following image:
[3D scene coordinate system]()

The `*.csv` file contain the following 10 columns:
0. frame_index: strart from 0;
1. camera_position_x: camera position
2. camera_position_y
3. camera_position_z
4. camera_rotation_x: camera orientation with euler angle degrees
5. camera_rotation_y
6. camera_rotation_z

Use the Blender project `traj_visual.blend` ues to visual the camera trajectory.
The load the scene `*.ply` file, to check the quality of trajectory.

## 2.2. Step 2: Render data

### 2.2.1. Render Raw data

Render the scene with camera trajectory, and generate 3 kinds of raw data: rgb, optical flow and single depht map.
The generated data output to the folder named with `data/$scene_name$_YYYY-MM-DD-HH-mm-ss`.

Run the render program twice with the trajectory file and center file separately.

The output file name:
- %04d_depth.bin: raw data float , The unit is millimeter.
- %04d_rgb.jpg:
- %04d_opticalflow_forward/backward.bin: raw data with x, y float for each element 
- centre_rgb.jpg: 
- cnetre_depth.bin: raw data with float for each element, The centre view depht map. The single depth map is the depht information as centre of the scene, The unit is millimeter.

The program is name with `ReplicaVideoRendererMP`, and the input parameters are (in order):
- mesh.ply: absoluate path of the dataset's `*.ply` file
- textures
- glass.sur[glass.sur/n]
- cameraPositions.txt[file.txt/n]
- spherical[y/n]
- outputDir
- width
- height

**Note** : the width should be the 2 time of the height.

And an example:
The output data will save on `/mnt/sda1/workspace_linux/replica360/data/$scene_name$_YYYY_MM_DD_HH_mm` folder.
The timestmp is the time when python script run.

```
/mnt/sda1/workspace_linux/replica360/build/ReplicaSDK/ReplicaVideoRendererMP \
/mnt/sda1/workdata/replica_v1_0/hotel_0/mesh.ply \
/mnt/sda1/workdata/replica_v1_0/hotel_0/textures/ \
/mnt/sda1/workdata/replica_v1_0/hotel_0/glass.sur \
/mnt/sda1/workspace_linux/replica360/glob/hotel_0_2020_05_10_15_21_25_circle[_center].csv \
y \
/mnt/sda1/workspace_linux/replica360/data/ \
960 \
480
```

### 2.2.2. Format Convertion

This step will:
- Convert the raw data to specified format;
- Generate the visualizion image of raw data.

The optical flow storage in the `.flo` file, and name with specifed extension `(original_name).flo`.
The rgb visualizion image  storage in `.jpg` files, and name with specifed extension `(original_name).jpg`.

This step will generate:
- center.obj: 
- centre_depth_visual.jpg:
- %04d_opticalflow_forward/backward.flo:
- %04d_opticalflow_forward/backward_visual.jpg:

The program is name with `DataFormatConversion`, and the input parameters are (in order):
- root_dir: the absoluate path of the previouse output directory.

Run the program with seprately input folder, one is the trajectory, another is center folder.


## 2.3. Step 3: Generate the MegaParallax Request Data

The python script `.py` use to rename the replic rendered file and megaparallax necessary files.
Move the nessary files to directory name with `_mp` postfix.

### 2.3.1. For MegaParallax `Preprocessing.exe`

The data output folder name `$dataset_name$_YY-MM-DD-HH-MM-SS_Preprocessing`.
Change the camera path from replica coordinate system to OpenVSLAM coordinate system.

- **frame_trajectory_with_filename.csv**: The camera trajectory path and obj file use the OpenVLSAM format. The camera tyrajectory path's time stamp compute with 1 FPS.

- **map.msg**: The `*.msg` file only storage the 3D point cloud.

- **cameras.txt**:

- **modelFiles.openvslam**:

- **rgb_image**: name with `panoramic-%04d.jpg`

### 2.3.2. For MegaParallax `Viewer.exe`

The data output folder name `$dataset_name$_YY-MM-DD-HH-MM-SS_Viewer`.

Put the data to Cache folder generate the folder in Cache 
The Cache folder named with `datasetName-GroundTruth`
The output file and directory structure as following:

- **Camera.csv**:
Storage the camera pose and cooresponding rgb image and flo files.

- **PreprocessingSetup-%04d.json**

- **PointCloud.csv**:
And the 3D mesh generated with the centre viewpoint depth map.
The `.obj` file name with `spherefit-disparity2depth-sm0.0-pr0.0.obj`

- **panoramic-%04d-FlowToPrevious.flo**:
The optical flow file follow the MegaParallax naming convention, `panoramic-%04d-FlowToNext.flo` and `panoramic-%04d-FlowToPrevious.flo`, e.g. `panoramic-0317-FlowToNext.flo`.

- **spherefit-depth-ground-truth.obj** :
The ground truth proxy geometry of sphere fitting.

More detail of the files reference the MegaParallax code.

# 3. FAQ


