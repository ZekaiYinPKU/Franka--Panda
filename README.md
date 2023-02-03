# Franka--Panda
这是静园五院两台franka-panda机械臂的简要使用说明
## 1、前言
1、
  本package是在陈德铭同学的package的基础上修改得到的，原文地址https://github.com/Derick317/Franka-Control

2、  
本package有非常多的前置系统要设置，包括实时内核、franka_ros、libfranka等，以及配套的realsense相关package，具体可参考

张继耀学长的https://github.com/Jiyao06/FrankaPanda

以及陈德铭同学的https://github.com/Derick317/Typora/blob/main/Softwares/Franka_and_ROS/set_up.md

进行安装，目前这个package可以在静园五院104控制机械臂的主机上使用。

3、
franka官方文档：

https://frankaemika.github.io/docs/overview.html

https://frankaemika.github.io/libfranka/

4、Panda机械臂软件上的小毛病一大堆，一定勤用timeshift。

## 2、Franka_Control
### 2.1、机械臂模块
这个package是在陈德铭同学开发的机械臂控制系统的基础上改写的，根本原因是Emika公司官方给的libfranka的在机械臂运动时拒绝其他线程读取机械臂的状态，所以就重新写了一份机械臂运动控制系统，并且将需要读取的状态重新存储在一个新的全局变量中。

以move_to_one_pos为例，整个package的使用逻辑如下：
  主程序（python）首先创建一个robot实例R
  ```
  #np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])是设置的robot初始状态
  r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)
  ```
  
  主程序调用franka.py中的函数set_next_goal()，将目标角度赋值给robot实例R的self.next_goal
  
  主程序调用franka.py中的set_next_goal_to_controller()，进而调用robotmodule.cpp中的set_next_goal_joints(double * joints)，将robot实例R的self.next_goal传给JointPosition_next。
  注：每个从python传给C的变量都需要使用franka.py中的convert_type函数，具体请学习ctype的使用。
  
  
  
最为关键的文件是/robot_arm/Franka-Control/robotmodule中的robotmodule.cpp以及/robot_arm/Franka-Control/franka.py，
### 2.2 夹爪模块
gripper模块目前实现的函数共有三个，分别是抓取、放开和重置，对应script中的gripper_close.py、gripper_open.py、和gripper_homing.py这部分比较简单，唯一值得一提的是gripper_close函数
```
import numpy as np
import sys
import time
sys.path.append('/home/hyperplane/Desktop/robot_arm/Franka-Control')
from franka import *
r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)
r.gripper_grasp(grasping_width = 0.035,  speed = 0.1,  force = 50,  epsilon_inner=0.005,  epsilon_outer=0.005)
```
gripper_grasp一共有5个参数，speed是抓取的速度、force是抓取使用的力，当抓取反馈的力达到force设定的值的时候将会停止抓取，此时如果两个夹爪的距离位于(grasping_width-epsilon_inner, grasping_width+epsilon_outer)时，在当前设置下为(0.03米,0.04米)之间，认为grasping成功并返回1，否则返回0。

### 2.3 抓取模块
scripts中的grasping.py是使用graspnet作为gripper_to_robot pose estimator的抓取程序，需要对camera进行标定后将transformation矩阵赋值到178行的robot_to_camera_transformation，然后就可以抓取了。
抓取的函数我已经写好集成在robotmodule.cpp第578行的grasp_object函数

### 2.4 其他辅助模块
#### 2.4.1 camera.py
使用这个函数可以打开realsense相机。

#### 2.4.2 record_joint.py
使用这个函数可以记录一段机械臂轨迹，可用于move_according_to_array或者连续的move_to_one_pos，核心函数为r.get_q_control()，它可以读取机械臂当前的角度信息。

#### 2.4.3 return_to_default_position.py
使用这个函数可以将机械臂回归到默认姿势。

## 6、机械臂使用方法、常见的硬件错误及应对方案
### 6.1、使用方法
0、机械臂由机械臂主体、机箱、红色按钮、灰色按钮和手持紧急停止开关组成，还有一根网线从机箱连接到电脑。

1、开机：先开白色桌子里的机箱开关，此时机械臂会变成黄灯，之后在浏览器登录172.16.0.2，在desk界面右侧打开锁定，机械臂会变为蓝灯。

2、关机：先在desk界面里shut down，等待3分钟左右desk会弹一个提示关闭机箱，此时再断电

3、重启：在desk界面reboot

注：无论是开机关机还是重启都非常耗时间，请耐心等待。

4、机械臂蓝灯模式为自动控制模式，白灯模式为手动示教模式，在示教模式中按下机械臂gripper上的两侧按钮可以手动移动机械臂。

5、在蓝灯模式下拖动机械臂会导致进入紫灯模式报错，此时需要按下灰色按钮进入白灯模式，再抬起灰色按钮恢复蓝灯模式。

6、在蓝灯模式下按下手持紧急停止开关同样会导致进入紫灯模式报错，此时需要按下灰色按钮进入白灯模式，再抬起灰色按钮恢复蓝灯模式。

7、在白灯模式下拖动机械臂如果关节运行到极限会进入红灯模式报错，此时需要重启机械臂（查看第3条）

### 6.2、常见错误及应对方案
1、名为gaitech的机械臂的红色按钮有时候会被卡住，具体表现为无论如何机械臂都无法开机，需要用力往上拔一下红色按钮。

2、名为gaitech的机械臂的gripper有的时候好像接触不良，具体表现为程序控制gripper合并但是机械臂没反应，通常情况下等1分钟左右即可，如果反复尝试都没反应，就用手合并一下gripper的夹爪。

3、名为franka的机械臂网络连接速度比gaitech慢，具体表现为打开172.16.0.2界面时会卡在connecting界面。一般情况下稍等片刻即可，如果一直connecting，则需要参考[franka官网](http://www.franka.cn/FCI/getting_started.html#setting-up-the-network)重新设置网络。

4、如果在运行程序时报错libfranka:······can not operate in this mode，无论机械臂是蓝灯还是白灯，都需要使用灰色按钮重置机械臂状态。
 
