import numpy as np
import sys
import time
sys.path.append('/home/hyperplane/Desktop/robot_arm/Franka-Control')
from franka import *
r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)
r.gripper_homing()
