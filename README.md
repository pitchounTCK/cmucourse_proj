# CMU Robot Autonomy Course Project (Team 4)

## Introduction
Using locobot to grasp a tool to write and draw  
1) Run the launch file `roslaunch locobot_control main.launch use_arm:=true torque_control:=true use_rviz:=false`  
2) In another terminal, load the pyrobot environment with `load_pyrobot_env`.  

`record.py`: Move by end effector position, saves and plot position  
`opengripper.py`: Convenient script to move arm, open and close gripper.  
`record_traj.py`: Records a human controlled trajectory for a skill. **Note**: use `torque_control:=false`  
`replay_traj.py`: Replays the recorded trajectory by end effector position.  
`DMP.py`: Fits a DMP from EE xyz position velocity and acceleration. 
Generates trajectory for the learned skill.  

Movement in x over 10 steps: ![Plot](https://github.com/eehantiming/cmucourse_proj/blob/master/tenloops.png "EE pose over time")  

## Resources
- [Discussion Document](https://docs.google.com/document/d/1hA72cqlCjKWrKFbTAD62o3VZzhTlJhqMhDQjm3AuSK4/edit#)  
- [Locobot docs](https://pyrobot-docs.readthedocs.io/en/latest/core/arm.html)  