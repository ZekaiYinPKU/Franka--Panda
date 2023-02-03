# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md
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
from lib.opts import opts
from lib.Dream_detector import DreamDetector
from lib.LM import *
import torch
from tqdm import tqdm
import dream_geo as dream
import dream as dream_original
from ruamel.yaml import YAML
import sys
import pyrealsense2 as rs
sys.path.append('/home/hyperplane/Desktop/zekaiyin/DREAM/Franka-Control')
sys.path.append('/home/hyperplane/Desktop/zekaiyin/grasp_transparent_object_hw/src/')
print(sys.path)
from franka import *
from policy import Policy
from graspnet_o.demo_centerformer import GraspNet_ol_opt
from graspnet_o.demo_centerformer_utils import *

from scipy.spatial.transform import Rotation as R

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def RMAT(rval, tval):
    rval = np.array(rval)
    tval = np.array(tval).reshape(3,1)
    ret = R.from_euler('xyz', rval,degrees = False)
    trans = np.concatenate((ret.as_matrix(),tval),axis = 1)
    trans = np.concatenate((trans, np.array([[0, 0, 0 , 1]])), axis=0)
    # trans = np.linalg.inv(trans)
    return trans




def generate_belief_map_visualizations(
    belief_maps, keypoint_projs_detected, keypoint_projs_gt=None
):

    belief_map_images = dream.image_proc.images_from_belief_maps(
        belief_maps, normalization_method=6
    )

    belief_map_images_kp = []
    for kp in range(len(keypoint_projs_detected)):
        if keypoint_projs_gt:
            keypoint_projs = [keypoint_projs_gt[kp], keypoint_projs_detected[kp]]
            color = ["green", "red"]
        else:
            keypoint_projs = [keypoint_projs_detected[kp]]
            color = "red"
        belief_map_image_kp = dream.image_proc.overlay_points_on_image(
            belief_map_images[kp],
            keypoint_projs,
            annotation_color_dot=color,
            annotation_color_text=color,
            point_diameter=4,
        )
        belief_map_images_kp.append(belief_map_image_kp)
    n_cols = int(math.ceil(len(keypoint_projs_detected) / 2.0))
    belief_maps_kp_mosaic = dream.image_proc.mosaic_images(
        belief_map_images_kp,
        rows=2,
        cols=n_cols,
        inner_padding_px=10,
        fill_color_rgb=(0, 0, 0),
    )
    return belief_maps_kp_mosaic

def get_ee_in_image_space(base_to_camera, ee_to_base, camera_intrinsic):
    ee_to_camera = base_to_camera @ ee_to_base
    ee_in_image_space = camera_intrinsic @ ee_to_camera[:3, :]
    ee_in_image_space /= ee_to_camera[2]
    return ee_in_image_space[0:2]


def get_ee_in_camera_space(base_to_camera, ee_to_base):
    ee_to_camera = base_to_camera @ ee_to_base
    return ee_to_camera[:3, :]


def get_ee_with_rotation(camera_to_robot_pose, T, camera_intrinsic, rotation):
    rot = np.eye(4)
    rot[0:3,0:3] = rotation
    Axiss = np.zeros((3,4))
    Axiss[0,0] += 1
    Axiss[1,1] += 1
    Axiss[2,2] += 1
    Axiss = rotation @ Axiss
    trans = np.repeat(T[0:3],4,axis = 1)
    Axiss += trans
    ee_in_camera_space =  camera_to_robot_pose[0:3,0:3] @ Axiss + camera_to_robot_pose[:,-1][:3].reshape(3,1)
    ee_in_image_space = camera_intrinsic @ ee_in_camera_space[:3, :]
    ee_in_image_space /= (ee_in_camera_space[2]+1e-8)
    return ee_in_image_space[0:2]


def get_3D_axis(base_to_camera, T, rotation):
    rot = np.eye(4)
    rot[0:3,0:3] = rotation
    Axiss = np.zeros((3,4))
    Axiss[0,0] += 1
    Axiss[1,1] += 1
    Axiss[2,2] += 1
    Axiss = rotation @ Axiss
    trans = np.repeat(T[0:3],4,axis = 1)
    Axiss += trans
    axis_in_camera_space =  base_to_camera[0:3,0:3] @ Axiss + base_to_camera[:,-1][:3].reshape(3,1)
    return axis_in_camera_space[:3, :]



def draw_ee(image, ee, size = 6):
    x = int(ee[0])
    y = int(ee[1])
    if x < 0 or x > 639 or y < 0 or y > 479:
        return image
    image = cv2.circle(image, (x, y), size, (255, 0, 0), -1)
    return image

def draw_arrow(image, center, end, color = (0, 0, 255)):
    x = int(center[0])
    y = int(center[1])
    end_x = int(end[0])
    end_y = int(end[1])

    len = ((end_y - y) ** 2 + (end_x - x) ** 2) ** 0.5
    if len == 0:
        end_x = x
        end_y = y
    else:
        end_x = int((end_x - x) / len * 40) + x
        end_y = int((end_y - y) / len * 40) + y
    h,w,_ = image.shape

    if x < 0 or x > w or y < 0 or y > h:
        return image
    cv2.arrowedLine(image, (x,y), (end_x,end_y), color, thickness = 2, line_type=cv2.LINE_AA, shift = 0, tipLength = 0.1)
    # r,g,b = color
    # image = ArrowDrow(image, x, y, end_x, end_y, r,g,b)
    return image

def draw_arrow_in_image(transform, ee, camera_K):
    ee_in_image_space = get_ee_in_image_space(transform, ee, camera_K,2)
    X,Y,Z,center = ee_in_image_space.T
    color_image = draw_ee(color_image, center,3)
    color_image = draw_arrow(color_image, center, X,(0,0,255))
    color_image = draw_arrow(color_image, center, Y,(0,255,0))
    color_image = draw_arrow(color_image, center, Z,(255,0,0))


def get_weights(x2d_input,x3d_input,transform,camera):
    num_points,_ = x2d_input.shape
    weights = np.zeros((num_points+1,2))

    for i in range(num_points):
        x2d_tmp = x2d_input[i]
        if(x2d_tmp[0] < -1000):
            continue
        x3,y3,z3 = x3d_input[i]
        x3d_tmp = np.array([x3,y3,z3,1])
        x2d_rep = camera @ transform[0:3] @ x3d_tmp
        x2d_rep[0]/=x2d_rep[2]
        x2d_rep[1]/=x2d_rep[2]
        x2d_rep = x2d_rep[0:2]
        dis = np.linalg.norm(x2d_rep - x2d_tmp)
        for j in range(2):
            weights[i,j] = np.exp(-5 * dis)
        
    weights[-1] = [1e8,1e8]
    return weights,num_points

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

class Camera():
    def __init__(self, fx, fy, cx, cy, xres, yres):
        self.xres = xres
        self.yres = yres
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    def print_info(self):
        print('fx: ', self.fx,
              'fy: ', self.fy,
              'cx: ', self.cx,
              'cy: ', self.cy,
              'xres: ', self.xres,
              'yres: ', self.yres)

class robot_pose():
    def __init__(self, x2d, x3d, joints, transforms, end_effector,rep_error):
        self.x2d = x2d[x2d[:,0]>0]
        self.x3d = x3d[x2d[:,0]>0]
        self.joints = joints
        self.transforms = transforms
        self.mean_dis = 0
        self.min_dis = 0
        self.end_effector = end_effector
        self.rep_error = rep_error
    def print_info(self):
        print("x2d: ",self.x2d, "x3d: ",self.x3d, "joints: ", self.joints, "end_effector",self.end_effector, sep = "\n")

    def __lt__(self, other):
        # obj < other
        return self.mean_dis < other.mean_dis

    def __le__(self, other):
        # obj <= other
        return self.mean_dis <= other.mean_dis

    def __eq__(self, other):
        # obj == other
        return self.mean_dis == other.mean_dis

    def __ne__(self, other):
        # obj != other
        return self.mean_dis != other.mean_dis

    def __gt__(self, other):
        # obj > other
        return self.mean_dis > other.mean_dis

    def __ge__(self, other):
        # obj >= other
        return self.mean_dis >= other.mean_dis





def create_point_cloud_from_depth_image(depth, camera, organized=True):
    h, w = depth.shape
    scale_w = w / camera.xres
    scale_h = h / camera.yres
    xmap = np.arange(w)
    ymap = np.arange(h)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - camera.cx * scale_w) * points_z / (camera.fx * scale_w)
    points_y = (ymap - camera.cy * scale_h) * points_z / (camera.fy * scale_h)
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud



def visualize_grasp(out,state,depth_intrinsics,rs,verts,texcoords,color_source,h,w):
    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10, state=state)
    frustum(out, rs, depth_intrinsics, state)
    axes(out, view([0, 0, 0], state), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, state, color_source[:, :, (2, 1, 0)])    # 435不需[:, :, (2, 1, 0)]交换通道
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, state, color_source[:, :, (2, 1, 0)])    # 435不需[:, :, (2, 1, 0)]交换通道
        tmp = cv2.resize(tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot, state), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))
    cv2.imshow(state.WIN_NAME, out)
    key2 = cv2.waitKey(1)

    if key2 == ord("r"):
        state.reset()

    if key2 == ord("p"):
        state.paused ^= True

    if key2 == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key2 == ord("z"):
        state.scale ^= True

    if key2 == ord("c"):
        state.color ^= True

    if key2 == ord("s"):
        timestamp = time.time()
        #print(state.decimate)

        #cv2.imwrite('./ply/%s_out.png' % timestamp, out)
        plt.imsave("/home/hyperplane/realsense/%s_color.png" % timestamp, color_image)
        #plt.imsave("./ply/%s_out_depth_orign.png" % timestamp, depth_image_orign)

        #depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
        #print(depth_image.mean())
        depth_image = np.array(depth_image/1000, dtype=np.float32)
        cv2.imwrite("/home/hyperplane/realsense/%s_depth_120.exr" % timestamp, depth_image)
        #plt.imsave("./ply/%s_out_depth.png" % timestamp, depth_image)

        #points_orign.export_to_ply('./ply/%s_orign_out.ply' % timestamp, mapped_frame_orign)
        #points.export_to_ply('./captured_data/%s_out.ply' % timestamp, mapped_frame)

        # IR图像获取
        #cv2.imwrite("./ply/%s_out_ir_left.png" % timestamp, ir_left_image)
        #cv2.imwrite("./ply/%s_out_ir_right.png" % timestamp, ir_right_image)

    if key2 == ord("q"):               
        return 1

def network_inference(opt):
    real_keypoint_names = ["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link6", "panda_link7", "panda_hand"]
    detector = DreamDetector(opt,real_keypoint_names, is_real=opt.is_real, is_ct=opt.is_ct)

    state = AppState()

    robot_poses = []
    robot_poses_change = []

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

    #
    colorizer = rs.colorizer()

    graspnet = GraspNet_ol_opt(opt)



    out = np.empty((h, w, 3), dtype=np.uint8)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 500.000)
    # Load in image

    
    r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)
    
    
    if opt.move == 0:
        r.read_state()   
    else:
        r.yzk_start_control()

    flag_of_whether_update_transformation = 1
    flag_of_whether_to_get_grasp_pose = 0
    flag_draw_ee = 1
    flag_after_grasp_pose = 0

    robot_to_camera_transformation = np.eye(4)
    calibration_joints = np.array([[0.007905059921506204,-0.05131952781910946,1.5128603785969958,-2.353684503513971,0.03226702460641633,0.8746085111438258,2.377049497514135],[-0.3498451763010655,1.0791226563753975,1.5730753966231252,-1.5158788646886714,-1.0762372090286678,1.6311209785143534,1.813443801153865],[0.058585662772287835,0.42040903850091443,0.6041791021865713,-2.5386638280109435,0.03166393877069278,3.1632311993776177,1.278902720845637],[-1.3073793540600687,-0.11668066246378933,1.0329367475007729,-1.80183737810747,-0.07502985158231523,1.7245348718961078,1.2000463871579847],[-1.8398303691388622,1.0234313142508844,1.1034765067182677,-2.1959444535873107,-0.07776074771061846,2.8530409495159628,1.1877237123617501],[-2.5652542119491084,1.4690606135598423,2.5969016759895984,-2.232254253103541,-0.07343754245638552,1.1174028376971052,1.026172258604732],[0.050836360906822635,-0.5240054157062138,1.1357393792206254,-1.8719990573002872,-0.10952962466743257,0.8873578783117497,-1.4039216445282847] ,[-0.00047377701846702965, -0.7855447937431189, 0.0003260311383163978, -2.3561892689822015, 0.000589521053350634, 1.5704794415504568, 0.7849731242977285]])
    calibration_joints_index = 0
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

    ###
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    to_reset = True
    pts_realsense = o3d.geometry.PointCloud()
    pts_coord = o3d.geometry.PointCloud()
    fork = o3d.geometry.TriangleMesh()

    axis_link0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_link2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_link3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_link4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_link6 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_link7 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_hand = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    axis_ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    vis.add_geometry(pts_realsense)
    vis.add_geometry(pts_coord)
    vis.add_geometry(fork)

    vis.add_geometry(axis_link0)
    vis.add_geometry(axis_link2)
    vis.add_geometry(axis_link3)
    vis.add_geometry(axis_link4)
    vis.add_geometry(axis_link6)
    vis.add_geometry(axis_link7)
    vis.add_geometry(axis_hand)
    vis.add_geometry(axis_ee)
    ###
    transform_multi = np.eye(4)
    try_time = 0


    deviation_cnt = 0

    while 1:   
        idx += 1
        if flag_of_whether_to_get_grasp_pose and r.get_grasp_flag() == 0:
            try_time = 0
            if opt.autograsp == 0:
                cv2.destroyAllWindows()
                cv2.namedWindow(state.WIN_NAME, 0)
                cv2.setMouseCallback(state.WIN_NAME, mouse_cb)
            while 1: 
                if flag_of_whether_update_transformation == 1:
                    print("PLEASE FIX UPDATE FIRST")
                    # break 
                    # pass
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

                    # fork.transform(gripper)
                    
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
                    
                    print(grasp_pose_joints)

                    # pre_T_g_b = list(pre_T_g_b)
                    # pre_T_g_b[0] = list(pre_T_g_b[0])
                    # pre_T_g_b[1] = list(pre_T_g_b[1])
                    # pre_T_g_b[0][2] += 0.1

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
                if opt.autograsp == 0:
                    if visualize_grasp(out,state,depth_intrinsics,rs,verts,texcoords,color_source,h,w) == 1:
                        break
                else:
                    r.set_grasp_goal(pre_grasp_pose_joints+grasp_pose_joints+after_grasp_pose_joints)
                    r.start_grasp()
                    try_time = 0
                    time.sleep(0.05)
                    break

            

            if opt.autograsp == 0:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                to_reset = True
                pts_realsense = o3d.geometry.PointCloud()
                pts_coord = o3d.geometry.PointCloud()

                axis_link0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                axis_link2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                axis_link3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                axis_link4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                axis_link6 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                axis_link7 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                axis_hand = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                axis_ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

                vis.add_geometry(pts_realsense)
                vis.add_geometry(pts_coord)

                vis.add_geometry(axis_link0)
                vis.add_geometry(axis_link2)
                vis.add_geometry(axis_link3)
                vis.add_geometry(axis_link4)
                vis.add_geometry(axis_link6)
                vis.add_geometry(axis_link7)
                vis.add_geometry(axis_hand)
                vis.add_geometry(axis_ee)
                flag_of_whether_to_get_grasp_pose = 0


        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)       

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_frame = spatial.process(depth_frame)
        # depth_frame = temporal.process(depth_frame)
        # depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        color_image_bgr = np.asanyarray(color_frame.get_data())
        color_image_rgb = np.asanyarray(color_frame.get_data())
        color_image_rgb = cv2.cvtColor(color_image_rgb,cv2.COLOR_RGB2BGR)
        color_image = cv2.cvtColor(color_image_bgr,cv2.COLOR_RGB2BGR)
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_color_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        color_intrin = rs.video_stream_profile(color_frame.profile).get_intrinsics()
        # print("intrin", color_intrin)

        camera = Camera(fx=color_intrin.fx,
                        fy=color_intrin.fy,
                        cx=color_intrin.ppx,
                        cy=color_intrin.ppy,
                        xres=color_image.shape[1],
                        yres=color_image.shape[0])
        depth_image[depth_image > 3000] = 0
        pts = create_point_cloud_from_depth_image(depth_image, camera, organized=False)
        # print(pts.shape)

        # camera.print_info()

        
        image_rgb_OrigInput_asPilImage = PILImage.fromarray(color_image.astype(np.uint8))
        # read Camera_K
        camera_K = np.array([[color_intrin.fx, 0, color_intrin.ppx],
                             [0, color_intrin.fy, color_intrin.ppy],
                             [0, 0, 1]])
        
        angle_1,angle_2,angle_3,angle_4,angle_5,angle_6,angle_7 = r.get_q_control()

        panda_link_0_to_base = RMAT([0.000,0.000,0.000],[0.000,0.000, 0.000])#base
        
        panda_link_1_to_base = panda_link_0_to_base @ RMAT([0.000, 0.000, angle_1], [0.000, 0.000, 0.333])
        panda_link_2_to_base = panda_link_1_to_base @ RMAT([-1.57079633e+00, angle_2, 0.000],[0.000, 0.000, 0.000])
        #link1和link2之间没有translation，但是有一个joint的rotation
        panda_link_3_to_base = panda_link_2_to_base @ RMAT([1.57079633e+00, -angle_3, 0.000],[0.000, -0.316, 0.000])
        panda_link_4_to_base = panda_link_3_to_base @ RMAT([-1.57079633e+00, 3.14159265358979 + angle_4, -3.14159265358979],[0.083, 0.000, 0.000])
        panda_link_5_to_base = panda_link_4_to_base @ RMAT([-1.57079633e+00, angle_5, 0.0000],[-0.083, 0.384, 0.000])
        panda_link_6_to_base = panda_link_5_to_base @ RMAT([1.57079633e+00, -angle_6, 0.000],[0.000, 0.000, 0.000])
        panda_link_7_to_base = panda_link_6_to_base @ RMAT([1.57079633e+00, -angle_7, 0.000],[0.088, 0.000, 0.000])

        panda_link_8_to_base = panda_link_7_to_base @ RMAT([0.000, 0.000, 0.000],[0.000, 0.000, 0.107])
        panda_hand_to_base = panda_link_8_to_base @ RMAT([0.000, 0.000, -0.785],[0.000, 0.000, 0.000])
        panda_ee_to_base = panda_hand_to_base @ RMAT([0.000, 0.000, 0.000],[0.000, 0.000, 0.128])


        # import x3d
        # ans = r.get_trans().reshape(9,4,4).transpose(0,2,1)
        # panda_link_0_to_base = np.eye(4)
        # panda_link_1_to_base = ans[0]
        # panda_link_2_to_base = ans[1]
        # panda_link_3_to_base = ans[2]
        # panda_link_4_to_base = ans[3]
        # panda_link_5_to_base = ans[4]
        # panda_link_6_to_base = ans[5]
        # panda_link_7_to_base = ans[6]
        # panda_link_8_to_base = ans[7]
        # panda_hand_to_base = panda_link_8_to_base @ RMAT([0.000, 0.000, -0.785],[0.000, 0.000, 0.000])
        # panda_ee_to_base = panda_hand_to_base @ RMAT([0.000, 0.000, 0.000],[0.000, 0.000, 0.128])

        panda_link0_x3d = panda_link_0_to_base[:, -1][:3].tolist()
        panda_link2_x3d = panda_link_2_to_base[:, -1][:3].tolist()
        panda_link3_x3d = panda_link_3_to_base[:, -1][:3].tolist()
        panda_link4_x3d = panda_link_4_to_base[:, -1][:3].tolist()
        panda_link6_x3d = panda_link_6_to_base[:, -1][:3].tolist()
        panda_link7_x3d = panda_link_7_to_base[:, -1][:3].tolist()
        panda_hand_x3d = panda_hand_to_base[:, -1][:3].tolist()
        panda_ee_x3d = panda_ee_to_base[:, -1][:3].tolist()
        x3d_list = [panda_link0_x3d, panda_link2_x3d, panda_link3_x3d, panda_link4_x3d, panda_link6_x3d, panda_link7_x3d, panda_hand_x3d]
        # print("x3d_list", x3d_list)
        transform_list = [panda_link_0_to_base,panda_link_1_to_base,panda_link_2_to_base,panda_link_3_to_base,panda_link_4_to_base,panda_link_5_to_base,panda_link_6_to_base,panda_link_7_to_base,panda_link_8_to_base,panda_hand_to_base,panda_ee_to_base]
        
        if not opt.is_ct:
            img = PILImage.fromarray(np.uint8(color_image))
            origin_size = img.size
            img_shrink_and_crop = dream.image_proc.preprocess_image(
                img, (opt.input_w, opt.input_h), "shrink-and-crop"
                )
            img = np.asarray(img_shrink_and_crop)
            ret, detected_kps_np, detected_hms_tensors, transform_predicted, flag,rep_error = detector.run(img, idx, x3d=x3d_list, is_final=True, camera_K = camera_K, origin_size=origin_size)
        else:
            ret, detected_kps_np, detected_hms_tensors, transform_predicted, flag,rep_error = detector.run(color_image_bgr, idx, x3d=x3d_list, is_final=True, camera_K = camera_K)
        
        # print("detected_kps_np", detected_kps_np)
        list_length = 30
        
        transform_single = copy.deepcopy(transform_predicted)
        for i in range(1):
            if(flag):
                #able to solve
                x3d_np = np.array(x3d_list[1:])
                x2d_np = detected_kps_np[1:]
                x3d_np_h = np.concatenate((x3d_np[x2d_np[:,0]>0],np.ones((x3d_np[x2d_np[:,0]>0].shape[0],1))), axis = 1).T
                x3d_rep = np.linalg.inv(transform_predicted) @ transform_multi @ x3d_np_h
                pose = robot_pose(detected_kps_np, np.array(x3d_list), np.array([angle_1,angle_2,angle_3,angle_4,angle_5,angle_6,angle_7]), np.array(transform_list),np.array(panda_ee_x3d),np.linalg.norm(x3d_np_h-x3d_rep))

                

                if len(robot_poses) == 0:
                    robot_poses.append(pose)
                    break
                dis = 0
                min_dis = 100
                for i in range(len(robot_poses)):
                    dis += np.linalg.norm(robot_poses[i].end_effector - panda_ee_x3d)
                    min_dis = min(min_dis, np.linalg.norm(robot_poses[i].end_effector - panda_ee_x3d))
                dis /= len(robot_poses)
                pose.mean_dis = dis
                pose.min_dis = min_dis



                tmp_flag = 1
                if(tmp_flag):
                    if len(robot_poses) < list_length:
                        robot_poses.append(pose)
                        robot_poses.sort()
                    else:
                        if(dis > robot_poses[0].mean_dis):
                            for j in range(1,len(robot_poses)):
                                robot_poses[j].mean_dis -= np.linalg.norm(robot_poses[j].end_effector - robot_poses[0].end_effector)/len(robot_poses)
                                robot_poses[j].mean_dis += np.linalg.norm(robot_poses[j].end_effector - panda_ee_x3d)/len(robot_poses)
                            robot_poses[0] = pose
                            robot_poses.sort()    
                                # for i in range(len(robot_poses)):

                # tmp_dis = np.linalg.norm(robot_poses[i].end_effector - panda_ee_x3d)
                # if(tmp_dis < 0.0001):
                #     if(robot_poses[i].rep_error > rep_error):
                #         robot_poses[i] = pose

        # for i in robot_poses:
        #     print(i.rep_error,i.end_effector,sep = ' ')
        # print("")   
        if len(robot_poses) > 3:
            x2d_list_multi_frame = np.concatenate([x.x2d for x in robot_poses], axis = 0)
            x3d_list_multi_frame = np.concatenate([x.x3d for x in robot_poses], axis = 0)
            (success, translation_vector, rotation_vector) = dream.geometric_vision.solve_pnp(x3d_list_multi_frame, x2d_list_multi_frame, camera_K)   
            if(success):
                rotM = rotation_vector.matrix33.tolist()
                transform_multi[0:3,0:3] = rotM
                transform_multi[0:3,-1] = translation_vector.reshape(-1)
                
                # transform = transform_multi
            if opt.refinement:
                trans_init = transform_multi[:,-1][:3].reshape(1,3)
                rotmat_init = transform_multi[:3,:3]
                quat_init = R.from_matrix(rotmat_init).as_quat().reshape(1,4)
        
                #transform np(4,4)
                weights, num_points = get_weights(x2d_list_multi_frame,x3d_list_multi_frame,transform_multi, camera_K)
                # print(num_points)
                quat_after,trans_after = register_GN_C(x2d_list_multi_frame, x3d_list_multi_frame, quat_init, trans_init, weights, camera_K, num_points)

                quat = torch.tensor(quat_after)
                T = torch.tensor(trans_after)
                if torch.isnan(quat).any() or torch.isnan(T).any():
                    #didn't solved
                    transform_after = transform_multi
                else:
                    rotmat_after = R.from_quat(quat_after).as_matrix()
                    trans_after = trans_after.reshape(3,1)
                    transform_after = np.concatenate((np.concatenate((rotmat_after, trans_after),axis=1),np.array([[0,0,0,1]])),axis = 0)
                    transform_multi = transform_after
        # if len(robot_poses) > 25:
        #     robot_to_camera_transformation = transform_multi
            
        if(flag):
            x3d_np = np.array(x3d_list[1:])
            x2d_np = detected_kps_np[1:]
            x3d_np_h = np.concatenate((x3d_np[x2d_np[:,0]>0],np.ones((x3d_np[x2d_np[:,0]>0].shape[0],1))), axis = 1).T
            x3d_rep = np.linalg.inv(transform_predicted) @ transform_multi @ x3d_np_h
            # print(deviation_cnt)
            if np.mean(np.linalg.norm(x3d_np_h-x3d_rep,axis = 0)) > 0.1:
                robot_poses_change.append(pose)
                deviation_cnt += 1
            else:
                deviation_cnt = 0
                robot_poses_change = []
            if(deviation_cnt == 6):
                robot_poses = [pose]
                # print("POSE CHANGED, PLEASE FIX TRANSFORMATIOcN AGAIN")
                deviation_cnt = 0
        # print(len(robot_poses_change), len(robot_poses))
        print(robot_to_camera_transformation)
            
        # print("transform_predicted", transform_single)         
        # print("transform_after", transform_multi) 
        # print("updated_transform", robot_to_camera_transformation)

        # robot_to_camera_transformation = copy.deepcopy(transform_multi)
        # print(convert_transformation_matrix_to_array(robot_to_camera_transformation))
        if flag_of_whether_update_transformation == 1:
            robot_to_camera_transformation = copy.deepcopy(transform_multi)
        else:
            print(convert_transformation_matrix_to_array(robot_to_camera_transformation))

        blended_array = []
        detected_hms_imgs = dream.image_proc.images_from_belief_maps(detected_hms_tensors, normalization_method=6)

        for n in range(len(detected_hms_imgs)):

            detected_hms_img = detected_hms_imgs[n]
            kp = detected_kps_np[n]
            fname = real_keypoint_names[n]

            blended = PILImage.blend(
                image_rgb_OrigInput_asPilImage, detected_hms_img, alpha=0.5
            )
            blended = dream.image_proc.overlay_points_on_image(
                    blended,
                    [kp],
                    [fname],
                    annotation_color_dot="red",
                    annotation_color_text="white",
                )
            blended_array.append(blended)

        n_cols = int(math.ceil(len(real_keypoint_names) / 2.0))
        belief_maps_with_kp_overlaid_mosaic = dream.image_proc.mosaic_images(
                blended_array, rows=2, cols=n_cols, fill_color_rgb=(0, 0, 0)
            )
        cv2.namedWindow('mosaic', 0)
        cv2.imshow('mosaic', np.array(belief_maps_with_kp_overlaid_mosaic))

        ee = np.array([panda_ee_to_base[:, 3]]).T
        arr1 = [1.0014477424638484, 0.3613440665089721, 0.6706650007045643]
        arr2 = [ -0.3530710368027997,-0.5206609684614312, 0.33062197708830937, 0.7035212201192076]

        # robot_to_camera_transformation = np.linalg.inv(convert_array_to_transformation_matrix(np.array([arr1, arr2])))
        # robot_to_camera_transformation = convert_array_to_t#ransformation_matrix(np.array([arr1, arr2]))

# translation: 
#   x: 1.0014477424638484
#   y: 0.3613440665089721
#   z: 0.6706650007045643
# rotation: 
#   x: -0.3530710368027997
#   y: -0.5206609684614312
#   z: 0.33062197708830937
#   w: 0.7035212201192076

        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        # print("transform_predicted", transform)
        print("transform", robot_to_camera_transformation)
        # print("hand-eye",robot_to_camera_transformation)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if(flag_draw_ee == 0):
            # draw blue point based on fixed transformation
            ee_in_image_space = get_ee_in_image_space(robot_to_camera_transformation, ee, camera_K)

            ###
            ee_in_camera_space = get_ee_in_camera_space(robot_to_camera_transformation, ee)


            color_image = draw_ee(color_image, ee_in_image_space)
        elif(flag_draw_ee == 1):
            # the blue point between the fingers of end-effector is drawn with transformation matrix updating, and axises are drawn from fixed transformation matrix 
            ee_in_image_space = get_ee_in_image_space(transform_predicted, ee, camera_K)

            ###
            ee_in_camera_space = get_ee_in_camera_space(robot_to_camera_transformation, ee)


            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            color_image = draw_ee(color_image, ee_in_image_space)
            for index, i in enumerate(transform_list):
                rotation = np.array(i[0:3,0:3])
                ptr = np.array([i[:, 3]]).T
                ptr_in_image_space = get_ee_with_rotation(robot_to_camera_transformation, ptr, camera_K, rotation)
                ptr_in_camera_space = get_3D_axis(robot_to_camera_transformation, ptr, rotation)


                X,Y,Z,center = ptr_in_image_space.T
                color_image = draw_arrow(color_image, center, X,(0,0,255))
                color_image = draw_arrow(color_image, center, Y,(0,255,0))
                color_image = draw_arrow(color_image, center, Z,(255,0,0))
        else:
            pass

        if 1:
            vis.get_render_option().point_size = 3  # 点云大小
            vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色
            # pts_coord.colors = o3d.utility.Vector3dVector([255, 255, 255])
            # colored_pts = np.concatenate((pts, color_image.reshape(-1, 3)), axis=1)
            pts_realsense.points = o3d.utility.Vector3dVector(pts/1000.)
            pts_realsense.colors = o3d.utility.Vector3dVector(color_image_rgb.reshape(-1, 3) / 255)


            # axis_link0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # axis_link2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # axis_link3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # axis_link4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # axis_link6 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # axis_link7 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # axis_hand = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            # axis_ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

            axis_link0.transform(robot_to_camera_transformation @ panda_link_0_to_base)
            axis_link2.transform(robot_to_camera_transformation @ panda_link_2_to_base)
            axis_link3.transform(robot_to_camera_transformation @ panda_link_3_to_base)
            axis_link4.transform(robot_to_camera_transformation @ panda_link_4_to_base)
            axis_link6.transform(robot_to_camera_transformation @ panda_link_6_to_base)
            axis_link7.transform(robot_to_camera_transformation @ panda_link_7_to_base)
            axis_hand.transform(robot_to_camera_transformation @ panda_hand_to_base)
            axis_ee.transform(robot_to_camera_transformation @ panda_ee_to_base)

            vis.update_geometry(pts_realsense)
            vis.update_geometry(pts_coord)
            vis.update_geometry(fork)

            vis.update_geometry(axis_link0)
            vis.update_geometry(axis_link2)
            vis.update_geometry(axis_link3)
            vis.update_geometry(axis_link4)
            vis.update_geometry(axis_link6)
            vis.update_geometry(axis_link7)
            vis.update_geometry(axis_hand)
            vis.update_geometry(axis_ee)

            if to_reset:
                vis.reset_view_point(True)
                to_reset = False
            vis.poll_events()
            vis.update_renderer()
            ###

            axis_link0.transform(np.linalg.inv((robot_to_camera_transformation @ panda_link_0_to_base)))
            axis_link2.transform(np.linalg.inv((robot_to_camera_transformation @ panda_link_2_to_base)))
            axis_link3.transform(np.linalg.inv((robot_to_camera_transformation @ panda_link_3_to_base)))
            axis_link4.transform(np.linalg.inv((robot_to_camera_transformation @ panda_link_4_to_base)))
            axis_link6.transform(np.linalg.inv((robot_to_camera_transformation @ panda_link_6_to_base)))
            axis_link7.transform(np.linalg.inv((robot_to_camera_transformation @ panda_link_7_to_base)))
            axis_hand.transform(np.linalg.inv((robot_to_camera_transformation @ panda_hand_to_base)))
            axis_ee.transform(np.linalg.inv((robot_to_camera_transformation @ panda_ee_to_base)))


        cv2.namedWindow('RealSense', 0)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        elif key == ord("u"):
            #whether to update transformation matrix
            flag_of_whether_update_transformation = not flag_of_whether_update_transformation
        
        elif key == ord("r"):
            #reset
            r.set_next_goal(reset_angle_list)
            r.set_next_goal_to_controller()
            r.start_control_one(8.0)

        
        elif key == ord("n"):
            r.set_next_goal(calibration_joints[calibration_joints_index])
            r.set_next_goal_to_controller()
            r.start_control_one()
            calibration_joints_index+=1
            if(calibration_joints_index == len(calibration_joints)):
                calibration_joints_index = 0
        
        elif key == ord("g"):
            if flag_of_whether_update_transformation == 1:
                print("please fix transformation first!!!")
                # continue
            flag_of_whether_to_get_grasp_pose = not flag_of_whether_to_get_grasp_pose


        
        elif key == ord("d"):
            if flag_draw_ee == 0:
                flag_draw_ee = 1
            elif flag_draw_ee == 1:
                flag_draw_ee = 2
            else:
                flag_draw_ee = 0
        
        elif key == ord("t"):
            r.set_next_goal(pre_grasp_pose_joints)
            r.set_next_goal_to_controller()
            r.start_control_one()
            i = input("PRESS ANY BUTTON TO PRECEED")
            r.set_next_goal(grasp_pose_joints)
            r.set_next_goal_to_controller()
            r.start_control_one(duration = 4.0)

        elif key == ord("f"):
            r.set_next_goal(pre_grasp_pose_joints)
            r.set_next_goal_to_controller()
            r.start_control_one(duration = 7.0)
            time.sleep(7.5)
            r.set_next_goal(grasp_pose_joints)
            r.set_next_goal_to_controller()
            r.start_control_one(duration = 4.0)
        
        # elif key == ord("i"):
        #     r.set_grasp_goal(pre_grasp_pose_joints+grasp_pose_joints)
        #     r.start_grasp()

        
        elif key == ord("c"):
            print(r.gripper_grasp())
        
        elif key == ord("h"):
            r.gripper_homing()
        
        elif key == ord("o"):
            print(r.gripper_open())



        elif key == ord("1"):
            r.set_next_goal([-1.0943588137417508,0.9342887385412858,1.1899914862970644,-1.1094775170945286,-0.8176335483410598,2.215094963232676,-1.2699805746980837])
            r.set_next_goal_to_controller()
            r.start_control_one()
        
        elif key == ord("2"):
            r.set_next_goal([-1.3991920873371269,0.776538732666579,1.195032465349164,-2.204303831669322,-0.8177433095276354,3.1410578818321224,-2.3892637374508707])
            r.set_next_goal_to_controller()
            r.start_control_one()
        
        elif key == ord("3"):
            r.set_next_goal([-0.06152035812014027,0.12739864130969786,1.108171454537587,-2.127421779097172,-0.9501520428955555,1.7738171572146886,-0.8264826530714828])
            r.set_next_goal_to_controller()
            r.start_control_one()
        
        elif key == ord("4"):
            r.set_next_goal([0.03621453537363559,0.828678712400893,0.3782747047240274,-1.0189339721077366,-0.3837221610179952,2.1252549743735782,-0.9465591766941877])
            r.set_next_goal_to_controller()
            r.start_control_one()
        
        elif key == ord("5"):
            r.set_next_goal([0.050836360906822635,-0.5240054157062138,1.1357393792206254,-1.8719990573002872,-0.10952962466743257,0.8873578783117497,-1.4039216445282847])
            r.set_next_goal_to_controller()
            r.start_control_one()
        
        elif key == ord("6"):
            r.set_next_goal([0.1680489365468945,0.8217062687340895,0.7041983770822224,-1.3665352994182818,-0.6140144143369463,3.05750609032313,0.9037661470464534])
            r.set_next_goal_to_controller()
            r.start_control_one()
        
        elif key == ord("7"):
            r.set_next_goal([-0.5068105909489747,-0.9614139458409838,0.5344779517692432,-1.4076938323249537,0.02188589263790214,0.8802388261991267,0.6289561034207709])
            r.set_next_goal_to_controller()
            r.start_control_one()

        elif key == ord("8"):
            r.set_next_goal([0.8857129455285649,-0.7156491021710352,0.1819819814348598,-2.369384852932215,1.5505487256810742,2.266411265850067,0.6151003284423214])
            r.set_next_goal_to_controller()
            r.start_control_one()

        elif key == ord("9"):
            r.set_next_goal([2.2102333965426997,-1.3585457389960485,-0.7349138513754682,-1.2113194409586374,-0.25608931997087264,1.6627387253434864,-1.0180149736822732])
            r.set_next_goal_to_controller()
            r.start_control_one(duration = 10.0)
        
        elif key == ord("0"):
            r.set_next_goal([-0.00047377701846702965, -0.7855447937431189, 0.0003260311383163978, -2.3561892689822015, 0.000589521053350634, 1.5704794415504568, 0.7849731242977285])
            r.set_next_goal_to_controller()
            r.start_control_one()







        if idx == 10000:
            break

'''
if __name__ == '__main__':
    c2rp = np.eye(4)
    ee_in_base = np.array([[1], 
                           [1],
                           [1],
                           [1]])
    camera_intrinsic = np.array([[50, 0, 25],
                                 [0, 50, 25],
                                 [0, 0, 1]])
    image = np.ones((200, 200))
    ee_in_image_space = get_ee_in_image_space(c2rp, ee_in_base, camera_intrinsic)
    print(ee_in_image_space)
'''

if __name__ == "__main__":
    opt = opts().init_infer(7)
    network_inference(opt)
