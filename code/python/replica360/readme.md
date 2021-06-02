# 1. Introduction

The code calls Replica-Dataset to render and post-process for synthetic dataset generation.
The dataset compose with panoramic RGB/depth map/optical flow.

**Feature**
- [x] Generate the rendering camera pose CSV file;
- [ ] Call Replica-Dataset to render the original data;
- [x] Stitch Replica-Dataset CubeMap RGB/DepthMap/OpticalFlow to panoramic image;
- [ ] Rename the Replica-Dataset panoramic file based on the dataset convention;
- [x] Generate the unavailable pixel mask;

## 1.1. Runtime Environment

Platform are Linux and Windows:
- Python
- Replica-Dataset: https://github.com/yuanmingze/Replica-Dataset

## 1.2. Dataset Scene

The Replica-Dataset has 18 scenes, some are not closure scene without ceiling or floor. 

| Scene Name             |closure|
| ---------------------- | ------|
|  apartment_0           |   Y   | 
|  apartment_1           |   N   |
|  apartment_2           |   N   |
|  frl_apartment_0       |   N   |
|  frl_apartment_1       |   N   |
|  frl_apartment_2       |   N   |
|  frl_apartment_3       |   N   |
|  frl_apartment_4       |   N   |
|  frl_apartment_5       |   N   |
|  hotel_0               |   Y   |
|  office_0              |   Y   |
|  office_1              |   Y   |
|  office_2              |   N   |
|  office_3              |   N   |
|  office_4              |   Y   |
|  room_0                |   Y   |
|  room_1                |   Y   |
|  room_2                |   Y   |

## 1.3. Convention

**Camera Pose Coordinate System**

Camera pose coordinate system is same as the Replica-Dataset coordinate system, Up is (0.0, 0.0, 1.0), forward is (1.0, 0.0, 0.0), left is (0.0, 1.0, 0.0).
The camera's initial position is (0.0, 0.0, 0.0) orientation is (0.0, 0.0, 0.0).

**Camera Pose File**

The trajectory file name with `datasetname_YYYY_MM_DD_HH_mm_[circle/square/line_path][_center].csv`.

The `*.csv` file contain the following columns:
1. frame_index: strart from 0;
2. camera_position_x: camera position
3. camera_position_y
4. camera_position_z
5. camera_rotation_x: camera orientation with euler angle degrees
6. camera_rotation_y
7. camera_rotation_z

**RGB/Depth/OpticalFlow Data**

The original file name convention please reference the Replica-Data project.
The code to stitch the Replica-Data cubemap data and convert the panoramic data.

Optical flow is without **wrap-around**, the optical flow warped pixel may beyond the image boundary, do not overflow to other side of image.

- Panoramic RGB Image:
    - {:04d}_rgb_pano.jpg: 

- Panoramic depth Map:
    - {:04d}_depth_pano.dpt: original depth map, unit is meter, unavailable pixels are less than 0.
    - {:04d}_depth_pano_visual.jpg:  

- Panoramic Optical Flow (Forward):
    - {:04d}_opticalflow_forward_pano.flo: The forward optical flow from current frame to next frame, without wrap-around.
    - {:04d}_opticalflow_forward_pano_visua.jpg:

- Panoramic Optical Flow (Backward):
    - {:04d}_opticalflow_backward_pano.flo: The backward optical flow from current frame to previous frame, without wrap-around.
    - {:04d}_opticalflow_backward_pano_visual.jpg: the visualized optical flow.

- Panoramic Optical mask:
    - {:04d}_mask_pano.png: the 8bit single channel png, unavailable pixel is 0.

# 2. Tutorial For Panoramic Optical Flow

The code generates the synthetic data for panoramic optical flow dataset.
The script prefix is "panoof_".

Windows 10 platform.

- Step 1: Generate the camera trajectory file for Replica Rending:
  - Create json camera path configuration file;
  - `gen_video_path_mp.py` to generate `csv` camera path file.
  
- Step 2: Render cubemap data:
  - To render cubemap images.

- Step 3: Generate the panoramic data: 
  - Run post processing to stich cubemap to panoramic images;
  - the output should include: dpt depth map, flo optical flow, jpg RGB image, png unavailable pixels mask;

# 3. Tutorial For OmniPhoto

The code generates the synthetic data for OmniPhoto quantitative evaluation.
The script prefix is "omniphoto_".

The post-process will generate two kinds of data.
One use for MegaParallalx `Preprocessing.exe` input, another use for the MegaParallax `Viewer.exe` input.

## 3.1. step 1: generate the camera trajectory file for Replica Rending

Use the file `megeparallax/gen_video_path_mp.py` to generate a `*.csv` file.
Before run the python script, change the `scene_name` and `nsteps`, e.t.c. value in the python.

- `scene_name`: the scene name is cooresponding the file `glob\$scene_name$.txt`;
- `nsteps`: how many frames will generate in the full circle;
- `radius`: the radius of the circle with unit is meter;
- `output_dir`: the trajectory file output folder;


The script will create to `*.csv` files.
One is the whole camera trajectory, another is the center position of the scene.

The camera coordinate system is as following image:
[3D scene coordinate system]()

Use the Blender project `traj_visual.blend` ues to visual the camera trajectory.
The load the scene `*.ply` file, to check the quality of trajectory.

## 3.2. Step 2: Render data

### 3.2.1. Render Raw data

Render the scene with camera trajectory, and generate 3 kinds of raw data: rgb, optical flow and single depht map.
The generated data output to the folder named with `data/$scene_name$_YYYY-MM-DD-HH-mm-ss`.

Run the render program twice with the trajectory file and center file separately.

The output file name:
- %04d_depth.bin: raw data float , The unit is millimeter.
- %04d_rgb.jpg:
- %04d_opticalflow_forward/backward.bin: raw data with x, y float for each element 
- center_rgb.jpg: 
- center_depth.bin: raw data with float for each element, The center view depht map. The single depth map is the depht information as centre of the scene, The unit is millimeter.

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
The timestamp is the time when python script run.

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

### 3.2.2. Format Convertion

This step will:
- Convert the raw data to specified format;
- Generate the visualizion image of raw data.

This step will generate:
- center.obj: 
- centre_depth_visual.jpg:
- %04d_opticalflow_forward/backward.flo:
- %04d_opticalflow_forward/backward_visual.jpg:

The program is name with `DataFormatConversion`, and the input parameters are (in order):
- root_dir: the absoluate path of the previouse output directory.

Run the program with seprately input folder, one is the trajectory, another is center folder.


## 3.3. Step 3: Generate the MegaParallax Request Data

The python script `.py` use to rename the replic rendered file and megaparallax necessary files.
Move the nessary files to directory name with `_mp` postfix.

### 3.3.1. For MegaParallax `Preprocessing.exe`

The data output folder name `$dataset_name$_YY-MM-DD-HH-MM-SS_Preprocessing`.
Change the camera path from replica coordinate system to OpenVSLAM coordinate system.

- **frame_trajectory_with_filename.csv**: The camera trajectory path and obj file use the OpenVLSAM format. The camera tyrajectory path's time stamp compute with 1 FPS.

- **map.msg**: The `*.msg` file only storage the 3D point cloud.

- **cameras.txt**:

- **modelFiles.openvslam**:

- **rgb_image**: name with `panoramic-%04d.jpg`

### 3.3.2. For MegaParallax `Viewer.exe`

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

# 4. FAQ


