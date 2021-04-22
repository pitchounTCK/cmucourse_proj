#### RUn Kinethsetics and then record IK 
from pyrobot import Robot
import time
import numpy as np
import matplotlib.pyplot as plt

arm_config=dict(control_mode = 'torque')
robot = Robot('locobot', arm_config=arm_config)

# Lists to save values for plotting.
ee_pos = []
current_pose = []
current_orientation = []
current_quat = []
Time=[]



target_torque=4*[0]
robot.arm.set_joint_torques(target_torque)

print("Go to Start position")
time.sleep(2)
print("3..")
time.sleep(1)
print("2..")
time.sleep(1)
print("1..")
time.sleep(1)
print("Starting to record")


# Parameters
STEPS = 1
TIME_BETWEEN_STEPS = 3

start = time.time() # Time since start, for plotting
lap = time.time() # Time since last step, for checking loop condition
for i in range(STEPS):
    # Save end effector positions.
	while time.time() - lap < TIME_BETWEEN_STEPS:
		elaptime = time.time() - start
		ee_pose = robot.arm.get_ee_pose(robot.arm.configs.ARM.ARM_BASE_FRAME)
		cur_pos, cur_ori, cur_quat = ee_pose
		current_pose.append(cur_pos)
		current_orientation.append(cur_ori)
		current_quat.append(cur_quat)
		Time.append(elaptime)
	lap = time.time()

print('Recording stopped')


# Plot EE pose over time.
y_labels = ['X', 'Y', 'Z']
fig,axes = plt.subplots(3)
for axis in range(3): # x y z
    axes[axis].plot(Time, [current_pose[i][axis] for i in range(len(current_pose))])
    axes[axis].set_xlabel('Time (s)')
    axes[axis].set_ylabel(f'{y_labels[axis]} (m)')
    axes[axis].annotate(str(current_pose[-1][axis]),xy=(3,current_pose[-1][axis]))

# axes[1].plot(Time, [ee_pos[i][1] for i in range(len(ee_pos))])
# axes[2].plot(Time, [ee_pos[i][2] for i in range(len(ee_pos))])
plt.show()

# eef_step=0.005
# displacement = displacement.reshape(-1, 1)

# path_len = np.linalg.norm(displacement)
# num_pts = int(np.ceil(path_len / float(eef_step)))
# if num_pts <= 1:
# 	num_pts = 2

# ##### Compute the IK based on the joint positions recorded
# total_positions = len(current_pose)
# for points in range(total_positions):
# 	waypoints_sp = np.linspace(0, path_len, num_pts)
# 	waypoints = current_pose[points] + waypoints_sp / float(path_len) * displacement
# 	way_joint_positions = []
# 	qinit = robot.arm.get_joint_angles().tolist()
# 	for i in range(waypoints.shape[1]):
# 		joint_positions = robot.arm.compute_ik(waypoints[:, i].flatten(),current_quat, qinit=qinit, numerical=numerical)
# 		if joint_positions is None:
# 			rospy.logerr('No IK solution found;','check if target_pose is valid')
# 			print("False")


##### Compute the IK based on the joint position recorded
import pdb; pdb.set_trace()
total_positions = len(current_pose)
saved_positions = []
for points in range(total_positions):
	joint_positions = robot.arm.compute_ik(current_pose[points], current_orientation[points], numerical=True)  ###self.compute_ik(position, orientation, numerical=numerical)
	# result = False
	if joint_positions is None:
		print('None!')
		# rospy.logerr('No IK solution found; check if target_pose is valid')
	else:
		saved_positions.append(joint_positions)

# import pickle
# with open('saved_positions.p', 'wb') as f:
# 	pickle.dump(saved_positions, f)
# 		# result = robot.arm.set_joint_positions(joint_positions, plan=plan, wait=wait)

