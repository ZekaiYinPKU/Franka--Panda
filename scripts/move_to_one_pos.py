import numpy as np
import sys
import time
sys.path.append('/home/hyperplane/Desktop/robot_arm/Franka-Control')
from franka import *
from policy import Policy
r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)

#   set_next_goal() will set the goal joint into a variable.
r.set_next_goal([-0.012850516780949475,-0.7838697842836208,0.013130240670280811,-2.3651562852692187,-0.013147315034667732,1.57067876291275,0.7788902073593602])

#   set_next_goal_to_controller() will send the goal in the variable to the controller.
r.set_next_goal_to_controller()

#   start_control_one will start move the robot, duration will be the time cost for the trajectory.
duration = 7.0
r.start_control_one(duration)

#   start_control_one() will create a new thread to control, while doing so, we need to sleep the main thread.
while(1):
    print(r.get_q_control())
