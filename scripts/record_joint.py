import cv2
import numpy as np
import sys
sys.path.append('/home/hyperplane/Desktop/robot_arm/Franka-Control')
from franka import *

file1 = "/home/hyperplane/Desktop/robot_arm/recorded_joints.txt"
with open(file1,'w+') as f:
    f.write('[') 
i = 0
r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)
r.read_state()
q = 0
while(1):
    i = input("PRESS ANY BUTTON TO RECORD POSE")
    if i == '0':
        break;
    lst2 = [str(x) for x in r.get_q_control().tolist()]
    q+=1
    print('frame: ',q)
    print(','.join(lst2))
    with open(file1,"a+")as f:
        f.write('[')
        f.write(','.join(lst2))
        f.write('],')

with open(file1,'a+') as f:
    f.write(']') 
	


