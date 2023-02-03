import numpy as np
import sys
import time
sys.path.append('/home/hyperplane/Desktop/robot_arm/Franka-Control')
from franka import *
from policy import Policy
from scipy.spatial.transform import Rotation as R
def RMAT(rval, tval):
    rval = np.array(rval)
    tval = np.array(tval).reshape(3,1)
    ret = R.from_euler('xyz', rval,degrees = False)
    trans = np.concatenate((ret.as_matrix(),tval),axis = 1)
    trans = np.concatenate((trans, np.array([[0, 0, 0 , 1]])), axis=0)
    # trans = np.linalg.inv(trans)
    return trans


r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)
r.start_control_array()

while(1):
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
    print(panda_ee_to_base)