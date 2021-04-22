import pickle
from pyrobot import Robot

with open('saved_positions.p', 'rb') as f:
	saved_positions = pickle.load(f)

robot = Robot('locobot')

for joint_position in saved_positions:
	robot.arm.set_joint_positions(joint_position, plan=True, wait=True)