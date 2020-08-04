# -*- coding: utf-8 -*-
# the code use to load MegaParallax generated Camera.csv file

import bpy
from colorsys import hsv_to_rgb
import mathutils
import math
from decimal import Decimal
constant_pi = 3.1415926

#mathutils.noise.seed_set(1)
openvslam_traj_file = "/mnt/sda1/workspace_linux/replica360/data/hotel_0_2020_05_03_15_14_03_circle.csv"

clean_scene_enable = True
camera_obj_size_scalar = 0.3
unit_scalar = 1

def pseudocolor(val, minval, maxval):
    """ Convert val in range minval..maxval to the range 0..120 degrees which
        correspond to the colors Red and Green in the HSV colorspace.
    """
    h = (float(val-minval) / (maxval-minval)) * 120.0

    # Convert hsv color (h,1,1) to its rgb equivalent.
    # Note: hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    r, g, b = hsv_to_rgb(h/360.0, 1., 1.)
    return r, g, b


def draw_coord_axis():
    """
    add camera add orientation
    """
    pass

def add_frustum():
    """
    add frustum for camera
    """
    pass

def draw_traj():
    """
    """

    """
    1.delete all objects
    """
    if clean_scene_enable:
        bpy.ops.object.select_all(action='DESELECT')# Deselect all
        for ob in bpy.context.scene.objects:
            ob.select_set(True)
            bpy.ops.object.delete()

    """
    2.add openvslam_camera model
    """
    # add openvslam model
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    # add camera's cone
    bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=2, enter_editmode=False, location=(0, 0, 0))
    cone = bpy.context.active_object
    bpy.context.object.rotation_euler[0] = 0.0
    bpy.context.object.rotation_euler[1] = 0.0
    bpy.context.object.rotation_euler[2] = 0.0
    bpy.context.object.location[0] = 0.0
    bpy.context.object.location[1] = 0.0
    bpy.context.object.location[2] = -1.0
    # add camera's cube
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))
    # grope them
    cube = bpy.context.active_object
    cube.select_set(True)
    cone.select_set(True)
    bpy.ops.object.join()
    openvslam_cam = bpy.context.active_object
    moving_camera_name = "moving_camera_{:04d}".format(int(mathutils.noise.random() * 100))
    openvslam_cam.name = moving_camera_name

    bpy.ops.object.select_all(action='DESELECT')
    openvslam_cam.select_set(True)
    bpy.ops.transform.resize(value=(camera_obj_size_scalar,camera_obj_size_scalar,camera_obj_size_scalar))

    # add material to openvslam_cam
    openvslam_cam = bpy.context.active_object
    material = bpy.data.materials.new(moving_camera_name)# Get material
    if material is None:
        material = bpy.data.materials.new(name="Material_cam")    # create material
        # Assign it to object# assign to 1st material slot # no slots
    if openvslam_cam.data.materials:
        openvslam_cam.data.materials[0] = material
    else:
        openvslam_cam.data.materials.append(material)

    # try/except if the setting is not as expected
    try:
        # Get its first material slot
        material.use_nodes = True
        # Get the nodes in the node tree
        nodes = material.node_tree.nodes
        # Get a principled node
        principled = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
        # Get the slot for 'base color'
        base_color = principled.inputs['Base Color'] #Or principled.inputs[0]
        # Get its default value (not the value from a possible link)
        value = base_color.default_value
        # Translate as color
        value[0]=mathutils.noise.random()
        value[1]=mathutils.noise.random()
        value[2]=mathutils.noise.random()
        print(value)
    except:
        print('not found')

    """
    3.add openvslam_camera key frame
    """

    # read file and add key frames
    f = open(openvslam_traj_file)
    frame_tra_all = f.read().splitlines()
    current_frame_idx = -1
    #video_fps = 50

    # get the start index
    start_frame_index = 999999
    for frame_tra in frame_tra_all:
        frame_tra_list = frame_tra.split(' ')
        current_frame_idx = int(frame_tra_list[0])#*video_fps
        if start_frame_index > current_frame_idx:
            start_frame_index = current_frame_idx

    # add key frame
    for frame_tra in frame_tra_all:

        frame_tra_list = frame_tra.split(' ')

        current_frame_idx = int(frame_tra_list[0])#*video_fps
        print("add camera index {}".format(current_frame_idx))

        # quat_a = mathutils.Quaternion((Decimal(frame_tra_list[7]),Decimal(frame_tra_list[4]),Decimal(frame_tra_list[5]),Decimal(frame_tra_list[6])))
        # euler_a = quat_a.to_euler('XYZ')

        openvslam_cam.rotation_euler.x = float(frame_tra_list[4]) / 180.0 * constant_pi
        openvslam_cam.rotation_euler.y = float(frame_tra_list[5]) / 180.0 * constant_pi - constant_pi / 2
        openvslam_cam.rotation_euler.z = float(frame_tra_list[6]) / 180.0 * constant_pi

        openvslam_cam.location.x = float(frame_tra_list[1]) * unit_scalar
        openvslam_cam.location.y = float(frame_tra_list[2]) * unit_scalar
        openvslam_cam.location.z = float(frame_tra_list[3]) * unit_scalar
  
        print("add keyframe index {}".format(current_frame_idx))
        openvslam_cam.keyframe_insert(data_path="location", frame = current_frame_idx, index=-1)
        openvslam_cam.keyframe_insert(data_path="rotation_euler", frame = current_frame_idx, index=-1)
        
        """
        add the cameras for each frame
        """
        bpy.ops.mesh.primitive_cone_add(radius1=0.4, radius2=0, depth=0.5, enter_editmode=False)

        # Get active object
        cam_obj = bpy.context.active_object
        cam_obj.name = "camera_{:04d}".format(int(current_frame_idx))
        
        cam_obj.rotation_euler.x = float(frame_tra_list[4]) / 180.0 * constant_pi 
        cam_obj.rotation_euler.y = float(frame_tra_list[5]) / 180.0 * constant_pi - constant_pi / 2
        cam_obj.rotation_euler.z = float(frame_tra_list[6]) / 180.0 * constant_pi 
        
        cam_obj.location.x = float(frame_tra_list[1]) * unit_scalar
        cam_obj.location.y = float(frame_tra_list[2]) * unit_scalar
        cam_obj.location.z = float(frame_tra_list[3]) * unit_scalar

        cam_obj.scale[0] = camera_obj_size_scalar
        cam_obj.scale[1] = camera_obj_size_scalar
        cam_obj.scale[2] = camera_obj_size_scalar

        material_name = "Mat_%i" % int(current_frame_idx)
        mat = bpy.data.materials.new(material_name)
        if mat is None:
            mat = bpy.data.materials.new(name=material_name)    # create material
        # Assign it to object# assign to 1st material slot # no slots
        if cam_obj.data.materials:
            cam_obj.data.materials[0] = mat
        else:
            cam_obj.data.materials.append(mat)
        
        # Get its first material slot
        #material = obj.material_slots[0].material
        # Enable 'Use nodes':
        mat.use_nodes = True
        material = mat
        # Get the nodes in the node tree
        nodes = material.node_tree.nodes
        # Get a principled node
        principled = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
        # Get the slot for 'base color'
        base_color = principled.inputs['Base Color'] #Or principled.inputs[0]
            # Get its default value (not the value from a possible link)
        value = base_color.default_value

        # Translate as color
        value[0], value[1],  value[2]  = pseudocolor(current_frame_idx - start_frame_index, 0, len(frame_tra_all))
    
        #print(value[0],value[1],value[2],   value[3])

    bpy.context.scene.frame_end = current_frame_idx + 1

if __name__ == "__main__":
    #print("hello")
    draw_traj()