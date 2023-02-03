from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from gettext import translation
import math
import os
import open3d as o3d
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image as PILImage

import numpy as np
from ruamel.yaml import YAML
import torch
import cv2
import torchvision.transforms as TVTransforms
import time
import pybullet as p
import pybullet_data as pd
# import tools._init_paths as _init_paths
import sys
import json
import copy
import torch
import sys
import pyrealsense2 as rs
sys.path.append('/home/hyperplane/Desktop/robot_arm/Franka-Control')
sys.path.append('/home/hyperplane/Desktop/zekaiyin/grasp_transparent_object_hw/src/')
print(sys.path)
from franka import *
from policy import Policy
from graspnet_o.demo_centerformer import GraspNet_ol_opt
from graspnet_o.demo_centerformer_utils import *
from scipy.spatial.transform import Rotation as R

def convert_transformation_matrix_to_array(transform_mat):

    trans = transform_mat[0:3,-1]
    rotation = transform_mat[0:3,0:3]

    quat = R.from_matrix(rotation).as_quat()

    T_c_r = np.array([[0.,0.,0.],[0.,0.,0.,0.]])
    T_c_r[0][0]=trans[0]
    T_c_r[0][1]=trans[1]
    T_c_r[0][2]=trans[2]
    T_c_r[1][0]=quat[0]
    T_c_r[1][1]=quat[1]
    T_c_r[1][2]=quat[2]
    T_c_r[1][3]=quat[3]
    return T_c_r

class opts(object):
    def __init__(self) -> None:
        self.checkpoint_path='/home/hyperplane/Desktop/zekaiyin/grasp_transparent_object_hw/src/graspnet_o/logs/checkpoint.tar'
        self.num_point=20000
        self.num_view=300
        self.collision_thresh=0.01
        self.voxel_size=0.01

def convert_array_to_transformation_matrix(gg):
    translation = np.zeros(3)
    rotation = np.zeros(4)
    translation[0]=gg[0][0]
    translation[1]=gg[0][1]
    translation[2]=gg[0][2]

    rotation[0]=gg[1][0]
    rotation[1]=gg[1][1]
    rotation[2]=gg[1][2]
    rotation[3]=gg[1][3]
    rot_mat = R.from_quat(rotation).as_matrix()

    transform_mat = np.eye(4)
    transform_mat[0:3,0:3] = rot_mat
    transform_mat[0:3,-1] = translation

    return transform_mat

def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = (360,640)
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.mean_dis -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.mean_dis -= dz

    state.prev_mouse = (x, y)



state = AppState()
pipeline = rs.pipeline()
rsconfig = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = rsconfig.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

#CENTERDREAM
# rsconfig.enable_stream(rs.stream.color, rs.format.rgb8, 30)
rsconfig.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)


#GRASPNET
rsconfig.enable_stream(rs.stream.depth,rs.format.z16, 30)
rsconfig.enable_stream(rs.stream.infrared, 1)
rsconfig.enable_stream(rs.stream.infrared, 2)


cfg = pipeline.start(rsconfig)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height
# PC
pc = rs.pointcloud()
# Decimation
state_decimate = 1
decimate = rs.decimation_filter()
# Depth to disparity
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
# Spatial:
spatial = rs.spatial_filter()
# Temporal:
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
colorizer = rs.colorizer()
opt = opts()
graspnet = GraspNet_ol_opt(opt)
out = np.empty((h, w, 3), dtype=np.uint8)
sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
sensor.set_option(rs.option.exposure, 500.000)
# Load in image


r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)


# r.read_state()   
### calibration
robot_to_camera_transformation = np.array([[-0.75547422 , 0.64263593 , 0.12758435 , 0.34323687], [ 0.04537325 , 0.2455822  ,-0.9683133  , 0.33309175], [-0.65360536 ,-0.72574682 ,-0.2146895  , 1.42272326], [ 0.    ,      0.    ,      0.      ,    1.        ]])

idx = 0
state = AppState()

grasp_pose_transform = np.eye(4)
grasp_pose_joints = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pd.getDataPath())
franka = p.loadURDF('franka_panda/panda.urdf')

reset_angle_list = [-0.00047377701846702965, -0.7855447937431189, 0.0003260311383163978, -2.3561892689822015, 0.000589521053350634, 1.5704794415504568, 0.7849731242977285] 
for i in range(7):
    p.resetJointState(franka,i,reset_angle_list[i])

transform_multi = np.eye(4)
try_time = 0


deviation_cnt = 0

while 1:   
    idx += 1
    if r.get_grasp_flag() == 0:
        try_time = 0
        while 1: 
            if not state.paused:
                frames = pipeline.wait_for_frames()
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
            
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                # IR图
                # ir_frame_left = frames.get_infrared_frame(1)
                # ir_frame_right = frames.get_infrared_frame(2)

                depth_frame_orign = depth_frame     ###
                # depth_frame = decimate.process(depth_frame)
                # depth_frame = depth_to_disparity.process(depth_frame)
                depth_frame = spatial.process(depth_frame)
                # depth_frame = temporal.process(depth_frame)
                # depth_frame = disparity_to_depth.process(depth_frame)
                depth_frame = hole_filling.process(depth_frame)
                depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
                w, h = depth_intrinsics.width, depth_intrinsics.height

                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image_orign = np.asanyarray(depth_frame_orign.get_data())     ###
                color_image = np.asanyarray(color_frame.get_data())
                # IR图
                # ir_left_image = np.asanyarray(ir_frame_left.get_data())
                # ir_right_image = np.asanyarray(ir_frame_right.get_data())

                depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

                if state.color:
                    mapped_frame, color_source = color_frame, color_image
                else:
                    mapped_frame, color_source = depth_frame, depth_colormap

                points = pc.calculate(depth_frame)
                pc.map_to(mapped_frame)

                ###
                depth_colormap_orign = np.asanyarray(colorizer.colorize(depth_frame_orign).get_data())
                if state.color:
                    mapped_frame_orign, color_source_orign = color_frame, color_image
                else:
                    mapped_frame_orign, color_source_orign = depth_frame_orign, depth_colormap_orign
                points_orign = pc.calculate(depth_frame_orign)
                pc.map_to(mapped_frame_orign)
                ###

                # Pointcloud data to arrays
                v, t = points.get_vertices(), points.get_texture_coordinates()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

                grasp_poses, grasp_pc, try_time = graspnet.get_grasppose(rgb=color_image, depth=depth_image,T_c_r = convert_transformation_matrix_to_array(np.linalg.inv(robot_to_camera_transformation)),try_time = try_time)
                print(try_time)
                if(try_time > 15):
                    print("NO OBJECT TO GRASP!!!")
                    break
                if grasp_poses[0] == -1:
                    continue
                
                T_g_b = convert_transformation_matrix_to_array( np.linalg.inv(robot_to_camera_transformation) @ convert_array_to_transformation_matrix(grasp_poses))
                
                back_matrix = np.eye(4)
                back_matrix[2][3] = -0.1
                pre_T_g_b = convert_transformation_matrix_to_array( convert_array_to_transformation_matrix(T_g_b) @ back_matrix)

                # 
                after_T_g_b = copy.deepcopy(T_g_b)
                after_T_g_b = list(after_T_g_b)
                after_T_g_b[0] = list(after_T_g_b[0])
                after_T_g_b[1] = list(after_T_g_b[1])
                after_T_g_b[0][2] += 0.15

                grasp_poses = grasp_poses[0] 
                print("pose: camera to gripper",grasp_poses)
                print("pose: base to hand", T_g_b)
                
                lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
                upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
                
                grasp_pose_joints = p.calculateInverseKinematics(franka,11, T_g_b[0], T_g_b[1],
                    lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                    upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                    jointRanges=[5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8],
                    restPoses=[-0.00047377701846702965, -0.7855447937431189, 0.0003260311383163978, -2.3561892689822015, 0.000589521053350634, 1.5704794415504568, 0.7849731242977285],
                    maxNumIterations=1000,  
                    residualThreshold=.001
                )
                grasp_pose_joints = list(grasp_pose_joints)
                
                for i in range(6):
                    while grasp_pose_joints[i] > upperLimits[i]:
                        grasp_pose_joints[i] -= math.pi * 2
                    while grasp_pose_joints[i] < lowerLimits[i]:
                        grasp_pose_joints[i] += math.pi * 2
                
                if  grasp_pose_joints[6] > upperLimits[6]-0.3:
                    grasp_pose_joints[6] -= math.pi
                
                if  grasp_pose_joints[6] < lowerLimits[6]+0.3:
                    grasp_pose_joints[6] += math.pi
                

                pre_grasp_pose_joints = p.calculateInverseKinematics(franka,11, pre_T_g_b[0], pre_T_g_b[1],
                    lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                    upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                    jointRanges=[5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8],
                    restPoses=[-0.00047377701846702965, -0.7855447937431189, 0.0003260311383163978, -2.3561892689822015, 0.000589521053350634, 1.5704794415504568, 0.7849731242977285],
                    maxNumIterations=1000,  
                    residualThreshold=.001
                )
            
                
                pre_grasp_pose_joints = list(pre_grasp_pose_joints)
                print(pre_grasp_pose_joints)
                for i in range(6):
                    while pre_grasp_pose_joints[i] > upperLimits[i]:
                        pre_grasp_pose_joints[i] -= math.pi * 2
                    while pre_grasp_pose_joints[i] < lowerLimits[i]:
                        pre_grasp_pose_joints[i] += math.pi * 2
                
                if pre_grasp_pose_joints[6] > upperLimits[6]-0.3:
                    pre_grasp_pose_joints[6] -= math.pi
                
                if pre_grasp_pose_joints[6] < lowerLimits[6]+0.3:
                    pre_grasp_pose_joints[6] += math.pi
                
                after_grasp_pose_joints = p.calculateInverseKinematics(franka,11, after_T_g_b[0], after_T_g_b[1],
                    lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                    upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                    jointRanges=[5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8],
                    restPoses=[-0.00047377701846702965, -0.7855447937431189, 0.0003260311383163978, -2.3561892689822015, 0.000589521053350634, 1.5704794415504568, 0.7849731242977285],
                    maxNumIterations=1000,  
                    residualThreshold=.001
                )
            
                
                after_grasp_pose_joints = list(after_grasp_pose_joints)
                print(after_grasp_pose_joints)
                for i in range(6):
                    while after_grasp_pose_joints[i] > upperLimits[i]:
                        after_grasp_pose_joints[i] -= math.pi * 2
                    while after_grasp_pose_joints[i] < lowerLimits[i]:
                        after_grasp_pose_joints[i] += math.pi * 2
                
                if after_grasp_pose_joints[6] > upperLimits[6]-0.3:
                    after_grasp_pose_joints[6] -= math.pi
                
                if after_grasp_pose_joints[6] < lowerLimits[6]+0.3:
                    after_grasp_pose_joints[6] += math.pi
                


            # Render
            r.set_grasp_goal(pre_grasp_pose_joints+grasp_pose_joints+after_grasp_pose_joints)
            r.start_grasp()
            try_time = 0
            time.sleep(0.05)
            break