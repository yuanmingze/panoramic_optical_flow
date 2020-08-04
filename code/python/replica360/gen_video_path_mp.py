import math
import csv
import sys
import numpy as np
from random import randint
from datetime import datetime
import pathlib
import math  


def create_camera_obj(center, camera_position_list, obj_file_path):
    """
    output the camera and center position to *.obj 
    and generate a 
    """
    # add center point, index is 1
    vec_str_list = "v {} {} {}\n".format(center[0],center[1],center[2])
    face_str_list = ""
    for idx, camera_position in enumerate(camera_position_list):
        # add vertex
        vec_str = "v {} {} {}\n".format(camera_position[1],camera_position[2],camera_position[3])
        vec_str_list = vec_str_list + vec_str
        # add face
        if idx > 0:
            face_str = "f 1 {} {}\n".format(idx + 1, idx + 2)
            face_str_list = face_str_list + face_str

    # face for last vertex
    face_str = "f 1 {} 2\n".format(len(camera_position_list) + 1)
    face_str_list = face_str_list + face_str

    obj_str = vec_str_list + face_str_list
    with open(obj_file_path, "wt") as obj_file:
        obj_file.write(obj_str)


def generate_circle_path_grid(scene_name, grid_size, radius, path_csv_file, \
                    center_point, lock_direction_enable=True):
    """
     the unit is Meter
     the circle in the plane of x_y
    """
    cx = center_point[0]
    cy = center_point[1]
    cz = center_point[2]

    radius_step = int(float(radius) / grid_size + 0.5) + 1
    x_min = cx - radius_step * grid_size
    x_max = cx + radius_step * grid_size
    y_min = cy - radius_step * grid_size
    y_max = cy + radius_step * grid_size

    navigable_positions = []

    counter = 0
    for x in np.arange(x_min, x_max + grid_size * 0.1, grid_size):
        for y in np.arange(y_min, y_max + grid_size * 0.1, grid_size):
            # skip the position out the circle
            if math.sqrt((x - cx) * (x - cx) + (y - cy) *(y - cy)) -  np.finfo(float).eps * 10  > radius:
                continue

            counter = counter + 1
            spot = [counter]

            # camera position
            position_x = x
            position_y = y
            position_z = cz
            spot += [position_x, position_y, position_z]

            # camera orientation (outward)
            if lock_direction_enable:
                raise ValueError("do not implement lock direction")
            else:
                spot += [0.0, 0.0, 0.0]

            navigable_positions.append(spot)

    # output camera pose file for render
    with open(path_csv_file,'w') as f:
        f.writelines(' '.join(str(j) for j in i) +'\n' for i in navigable_positions)
    print("output path file {}".format(path_csv_file))

    # output camera position obj file
    path_obj_file = path_csv_file + ".obj"
    create_camera_obj([cx,cy,cz + 0.07], navigable_positions, path_obj_file)
    print("output path 3D model file {}".format(path_obj_file))


def gen_circle_path(scene_name, steps, radius, path_csv_file, center_csv_file, \
                    initial_rotation, center_point=None, lock_direction_enable=False):
    '''
    Args:
        scene_name: 
        steps: the number of frame
        path_csv_file: the csv file store the postion and orientation
        center_csv_file: store the center of camera
        center_point: list of xyz
    '''
    if center_point is None:
        # open position file
        #samples a random valid starting position
        scene_centre_samples_file = "../glob/" + scene_name + ".txt"
        print("select center point from file {}".format(scene_centre_samples_file))
        scenePos = open(scene_centre_samples_file, "r")
        data = [[float(i) for i in line.split()] for line in scenePos]

        idx = randint(0, len(data))
        idx =221
        cx = data[idx][0]
        cy = data[idx][1]
        cz = data[idx][2] + 0.5
    else:
        cx = center_point[0]
        cy = center_point[1]
        cz = center_point[2]

    print("Sampled center position:", cx, cy, cz)
    # output the camera centre csv file
    with open(center_csv_file,'w') as f:
        f.writelines("0 {} {} {} {} {} {} \n".format(cx,cy, cz, 0.0, 0.0, 0.0))
    print("output centre file {}".format(center_csv_file))

    # generate and output the camera path csv file
    navigable_positions = []
    for i in range(steps):
        spot = [i]

        rad = float(i) / steps * math.pi * 2

        # camera position
        position_x = cx + math.cos(rad)*radius
        position_y = cy + math.sin(rad)*radius
        position_z = cz
        spot += [position_x, position_y, position_z]

        # camera orientation (outward)
        if lock_direction_enable:
            camera_rotation_x = initial_rotation[0]
            camera_rotation_y = initial_rotation[1]
            camera_rotation_z = rad / math.pi * 180
            spot += [camera_rotation_x, camera_rotation_y, camera_rotation_z]
        else:
            spot += [0, 0, 0]

        # append the data
        navigable_positions.append(spot)

    # output camera pose file for render
    with open(path_csv_file,'w') as f:
        f.writelines(' '.join(str(j) for j in i) +'\n' for i in navigable_positions)
    print("output path file {}".format(path_csv_file))

    # output camera position obj file
    path_obj_file = path_csv_file + ".obj"
    create_camera_obj([cx,cy,cz], navigable_positions, path_obj_file)
    print("output path 3D model file {}".format(path_obj_file))


def generate_path(root_dir, config):
    """
    @return: path_csv_file, center_csv_file
    """
    scene_name = config["scene_name"]
    nsteps = config["camera_traj"]["step_number"] 
    radius = config["camera_traj"]["radius"]
    center_point_position = [ \
        config["camera_traj"]["start_position"]["x"], \
        config["camera_traj"]["start_position"]["y"], \
        config["camera_traj"]["start_position"]["z"]]
    center_point_rotation = [\
        config["camera_traj"]["start_orientations"]["x"], \
        config["camera_traj"]["start_orientations"]["y"], \
        config["camera_traj"]["start_orientations"]["z"]]

    lock_direction_enable = config["camera_traj"]["lock_direction"]
    path_type = config["camera_traj"]["type"] 

    print("generate camera path for {}, view number is {}, radius is {}.".format(scene_name, nsteps, radius))
    print("center position is {}, rotation is {}.".format(center_point_position, center_point_rotation))

    if path_type == "circle":
        # # load center point, radius and step
        # input_center_filename = root_dir + "/circle_center.csv"
        # with open(input_center_filename, newline='') as csvfile:
        #     data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        #     #next(data, None)  # skip the headers
        #     for row in data:
        #         if len(row) ==  0 or row[0] == '#':
        #             continue
        #         scene_name = row[0]
        #         nsteps = int(row[1])
        #         radius = float(row[2])
        #         center_point_position = [float(row[3]), float(row[4]), float(row[5])]
        #         center_point_rotation = [float(row[6]), float(row[7]), float(row[8])]
        #print("load camera infor from {}".format(input_center_filename))
        # generate path file
        # output_filename = str(nsteps) + "_" + str(radius) + "_circle.csv"
        # output_center_filename = str(nsteps) + "_" + str(radius) + "_circle_center.csv"
        output_filename = "circle.csv"
        output_center_filename = "circle_center.csv"

        path_csv_file = root_dir + "/" + output_filename
        center_csv_file = root_dir + "/" + output_center_filename

        gen_circle_path(scene_name, nsteps, radius, \
                        path_csv_file, center_csv_file, center_point = center_point_position, \
                        initial_rotation = center_point_rotation, \
                        lock_direction_enable = lock_direction_enable)
        
        return path_csv_file, center_csv_file

    elif path_type == "grid":
        output_filename = "grid.csv"
        grid_size = config["camera_traj"]["grid_size"] 
        path_csv_file = root_dir + "/" + output_filename
        generate_circle_path_grid(scene_name, grid_size, radius, path_csv_file, \
                    center_point = center_point_position, lock_direction_enable= lock_direction_enable)
        
        return path_csv_file, None

    else:
        raise RuntimeError("Juse generate circle path")


if __name__ == '__main__':
    """
        #time_str = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")
    """
    scene_name = "room_0"
    root_dir = "/mnt/sda1/workdata/lightfield/GT-Replica/"
    generate_path(root_dir, scene_name)
