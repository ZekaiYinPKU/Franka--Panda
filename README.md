# Franka-Control
## 1、前言
1、
  本package是在陈德铭同学的package的基础上修改得到的，原文地址https://github.com/Derick317/Franka-Control， 感谢陈德铭同学的大力支持和帮助。

2、本package有非常多的前置依赖需要设置，包括实时内核、franka_ros、libfranka等，以及配套的realsense相关package，具体可参考

张继耀学长的https://github.com/Jiyao06/FrankaPanda

以及陈德铭同学的https://github.com/Derick317/Typora/blob/main/Softwares/Franka_and_ROS/set_up.md

进行安装，目前这个package可以在静园五院104控制机械臂的主机上使用，scripts文件夹中的所有程序均可python3运行。

3、
franka官方文档：

https://frankaemika.github.io/docs/overview.html

https://frankaemika.github.io/libfranka/

4、Panda机械臂软件上的小毛病一大堆，尤其是安装前置依赖的过程中，并且有几率把ubuntu系统直接搞崩，一定勤用timeshift备份，一定勤用timeshift备份，一定勤用timeshift备份。

## 2、Franka_Control

这个package是在陈德铭同学开发的机械臂控制系统的基础上改写的，最为关键的文件是/robot_arm/Franka-Control/robotmodule中的robotmodule.cpp以及/robot_arm/Franka-Control/franka.py。开发及改写的根本原因是Emika公司官方给的libfranka的在机械臂运动时拒绝其他线程读取机械臂的状态，导致无法获得机械臂实时的信息，所以就重新写了一份机械臂运动控制系统，并且将需要读取的状态重新存储在一个新的全局变量中。

每次修改robotmodule.cpp中的程序后，需要运行
```
sh /robot_arm/Franka-Control/buildmodule.sh
```
重新编译package。

如果移动了整个package的位置，则会编译报错为CMakeCache.txt记录的不一致，此时需要删除/robot_arm/Franka-Control/build文件夹中的所有内容重新编译

### 2.1、机械臂模块
在scripts里共有两个和基本控制相关的脚本，move_according_to_array.py和move_to_one_pos.py，前者会操作机械臂在几个不同的位置之间循环移动，后者会操作机械臂仅运行到下一个位置。

以move_to_one_pos为例，机械臂控制程序的运行逻辑如下：

  主程序(line 7)首先创建一个robot实例R。
  ```
  #np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])是设置的robot初始状态
  r = Robot("172.16.0.2",np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]), 0.03)
  ```

  主程序(line 10)调用franka.py中的函数set_next_goal()，将目标角度赋值给robot实例R的self.next_goal
  ```
  #franka.py line 211
  def set_next_goal(self,joints):
     self.next_goal = joints
  ```
  
  主程序(line 13)调用franka.py中的set_next_goal_to_controller()，进而调用robotmodule.cpp中的set_next_goal_joints(double * joints)，将robot实例R的self.next_goal传给JointPosition_next。
  ```
  #franka.py line 214
  def set_next_goal_to_controller(self):
    lib.set_next_goal_joints(convert_type(list(self.next_goal)))

  //robotmodule.cpp line 478
  void set_next_goal_joints(double * joints)
  {
      for (int i = 0; i < 7; i++)
      {
          JointPosition_next[i] = joints[i];
      }
  }
  ```
 
  注：每个从python传给C的变量都需要使用franka.py中的convert_type函数，**具体请学习ctype的使用**。
  
  主程序(line 17)调用franka.py中的start_control_one(duration = 7.0)，进一步调用robotmodule.cpp中的start_control_one(), 
  robotmodule.cpp中的start_control_one()会**创建一个新的线程**执行robotmodule.cpp中的control_one()函数，机械臂开始往下一个目标角度移动，duration为这段路径所需要的时间。
  这也是为什么主程序中line 20我们需要将主程序陷入死循环，因为control_one()是在一个新线程中运行的，如果主程序结束了，那么control_one()也将随之中断。
  ```
  #franka.py line 220
  def start_control_one(self, duration = 7.0):
    lib.start_control_one(convert_type(self.obj), convert_type(self.gripper), convert_type(duration))
   
  //robotmodule.cpp line 486
  void start_control_one(franka::Robot *robot,franka::Gripper * gripper, double duration){

    flag_reading = 0;
    if(flag_control == 0){
        flag_control = 1; 
        basic_data * data = new basic_data;
        data->robot = robot;
        data->gripper = gripper;
        data->duration = duration;

        robot->setCollisionBehavior(
                {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
                {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
        
        pthread_t tids[1];
        //创建一个新的线程
        pthread_create(&tids[0], NULL, control_one, data);       
    }
    else{
        std::cout<<"another control moving!"<<std::endl;
    }
   
}
    
```

在新的线程中，会调用control_one()函数，其他manipulation相关函数和这个差不多，我们目前采用的是基于角度进行控制，请阅读[libfranka官方文档](https://frankaemika.github.io/libfranka/classfranka_1_1Robot.html#a5b5ba0a4f2bfd20be963b05622e629e1)中关于control的部分

```
void * control_one(void* args){
    basic_data * data = (basic_data*)args;
    franka::Robot * robot = data->robot;
    franka::Gripper *gripper =data->gripper;
    
    //速度
    double duration = data->duration;
    

    std::array<double, 7> start_joints;
    std::array<double, 7> end_joints;

    double time = 0.0;
    
    
    //刚刚给过来的JointPosition_next。
    
    
    end_joints = JointPosition_next;
    
    //回调函数，可以理解为robot->control()函数的参数是个正在运行的函数
    //当时间还没有结束的时候，它的返回值为new_joints, robot->control()函数根据new_joints的值执行下一步的动作。
    //当时间结束后，它的返回值为franka::MotionFinished((franka::JointPositions)robot_state.q)，动作停止。
    
    robot->control([&time, &start_joints, &end_joints, &duration](const franka::RobotState& robot_state,
                                            franka::Duration period) -> franka::JointPositions  { 
        if (time == 0.0) 
            start_joints = CurrentRobotState.q;

        time += period.toSec();
        std::array<double, 7> new_joints;
        if (time <= duration)
        {
            new_joints = start_joints;
            for(int b = 0; b<7; b++){ 
                //指定下一个时刻new_joints的返回值，这里使用的是三角函数速度控制
                new_joints[b] += std::sin(M_PI * time / duration / 2) * std::sin(M_PI * time / duration / 2)  * (end_joints[b] - start_joints[b]);
            }
            
            //所有的get_系列函数（franka.py的125行开始）均读取CurrentRobotState或者CurrentRobotState_control
            //CurrentRobotState是所有的全局信息，CurrentRobotState_control是为了在不影响其他全局信息而获得实时信息新加的
            
            CurrentRobotState_control = robot_state;
            
        }
        else{
            CurrentRobotState.q = end_joints;
            return franka::MotionFinished((franka::JointPositions)robot_state.q);
        }

        return new_joints;
        });
    flag_reading = 1;
    flag_control = 0;
    return nullptr;
}
```   
**理解这部分的运行对于理解使用libfranka控制机械臂非常重要。**
    
### 2.2 夹爪模块
gripper模块目前实现的函数共有三个，分别是抓取、放开和重置，对应script中的gripper_close.py、gripper_open.py、和gripper_homing.py。

这部分比较简单，唯一值得一提的是gripper_close函数
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
scripts中的grasping.py是使用graspnet作为gripper_to_robot pose estimator的抓取程序，需要对camera进行标定后将transformation矩阵赋值到178行的robot_to_camera_transformation，然后就可以抓取了，程序会自动计算抓取的轨迹和路径。逆向运动学（从gripper的姿态求joint）使用的是pybullet进行求解。
抓取的函数我已经写好集成在robotmodule.cpp第578行的grasp_object函数。

### 2.4 其他辅助模块
#### 2.4.1 camera.py
使用这个函数可以打开realsense相机。

#### 2.4.2 record_joint.py
使用这个函数可以记录一段机械臂轨迹，可用于move_according_to_array或者连续的move_to_one_pos，核心函数为r.get_q_control()，它可以读取机械臂当前的角度信息。

#### 2.4.3 return_to_default_position.py
使用这个函数可以将机械臂回归到默认姿势。

## 3、机械臂使用方法、常见的硬件错误及应对方案
### 3.1、使用方法
0、机械臂由机械臂主体、机箱、红色按钮、灰色按钮和手持紧急停止开关组成，还有一根网线从机箱连接到电脑。

1、开机：先开白色桌子里的机箱开关，此时机械臂会变成黄灯，之后在浏览器登录172.16.0.2，在desk界面右侧打开锁定，机械臂会变为蓝灯。

2、关机：先在desk界面里shut down，等待3分钟左右desk会弹一个提示关闭机箱，此时再断电。**切勿直接断电。**

3、重启：在desk界面reboot

注：无论是开机关机还是重启都非常耗时间，请耐心等待。

4、机械臂蓝灯模式为自动控制模式，白灯模式为手动示教模式，在示教模式中按下机械臂gripper上的两侧按钮可以手动移动机械臂。

5、在蓝灯模式下拖动机械臂会导致进入紫灯模式报错，此时需要按下灰色按钮进入白灯模式，再抬起灰色按钮恢复蓝灯模式。

6、在蓝灯模式下按下手持紧急停止开关同样会导致进入紫灯模式报错，此时需要按下灰色按钮进入白灯模式，再抬起灰色按钮恢复蓝灯模式。

7、在白灯模式下拖动机械臂如果关节运行到极限会进入红灯模式报错，此时需要重启机械臂（查看第3条）

### 3.2、简单示教模式
franka也可以通过172.16.0.2/desk进行简单控制，在这个模式下无法读取内部数据，也无法运行其他程序抢占机械臂控制权。但是由于gripper和robot在运行逻辑中是两个不同的实例，可以运行夹爪相关的程序控制夹爪合并。

### 3.3、常见错误及应对方案
1、名为gaitech的机械臂的红色按钮有时候会被卡住，具体表现为无论如何机械臂都无法开机，需要用力往上拔一下红色按钮。

2、名为gaitech的机械臂的gripper有的时候好像接触不良，具体表现为程序控制gripper合并但是机械臂没反应，通常情况下等1分钟左右即可，如果反复尝试都没反应，就用手合并一下gripper的夹爪。

3、名为franka的机械臂网络连接速度比gaitech慢，具体表现为打开172.16.0.2界面时会卡在connecting界面。一般情况下稍等片刻即可，如果一直connecting，则需要参考[franka官网](http://www.franka.cn/FCI/getting_started.html#setting-up-the-network)重新设置网络。

4、如果在运行程序时报错libfranka:······can not operate in this mode，无论机械臂是蓝灯还是白灯，都需要使用灰色按钮重置机械臂状态。
 
