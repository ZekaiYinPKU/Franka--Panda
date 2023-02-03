import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

pipeline = rs.pipeline()
# rsconfig = rs.config()

config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(device)

config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)   # bgr8

# IR图
config.enable_stream(rs.stream.infrared, 1)
config.enable_stream(rs.stream.infrared, 2)

# Start streaming
pipeline.start(config)

sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
sensor.set_option(rs.option.exposure, 500.000)




# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# # Processing blocks
# pc = rs.pointcloud()
# decimate = rs.decimation_filter()
# decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
# colorizer = rs.colorizer()
#### Processing blocks ####
# PC
# pc = rs.pointcloud()
# Decimation
# state_decimate = 1

decimate = rs.decimation_filter()
# decimate.set_option(rs.option.filter_magnitude, 2 ** state_decimate)

# Depth to disparity
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# Spatial:
spatial = rs.spatial_filter()
#### spatial.set_option(rs.option.holes_fill, 0)  # between 0 and 5 def = 0
# spatial.set_option(rs.option.filter_magnitude, 2)  # between 1 and 5 def=2
# spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # between 0.25 and 1 def=0.5
# spatial.set_option(rs.option.filter_smooth_delta, 20)  # between 1 and 50 def=20

# Temporal:
temporal = rs.temporal_filter()
# temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
# temporal.set_option(rs.option.filter_smooth_delta, 20)

hole_filling = rs.hole_filling_filter()
colorizer = rs.colorizer()


while(1):
    # r.move_to_joint_position([-4.94010060e-04, -7.85622711e-01, 2.93664763e-04, -2.35617980e+00, 7.27985178e-04, 1.57059122e+00, 0]) # move robot to the specified pose
    
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    
    # Post processing
#    depth_frame = decimate.process(depth_frame)
    depth_frame = depth_to_disparity.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    # depth_frame = disparity_to_depth.process(depth_frame).as_frameset()
    depth_frame = disparity_to_depth.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    # # IR图
    # # ir_frame_left = frames.get_infrared_frame(1)
    # # ir_frame_right = frames.get_infrared_frame(2)

    depth_frame_orign = depth_frame     ###
    # depth_frame = decimate.process(depth_frame)

    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image_orign = np.asanyarray(depth_frame_orign.get_data())     ###
    
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('RealSense', 0)
    cv2.imshow('RealSense', color_image)
    cv2.namedWindow('RealSensed', 0)
    cv2.imshow('RealSensed', (100*depth_image))
    cv2.waitKey(1)
