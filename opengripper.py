'''
Use this script to insert/ remove tool from the robot gripper. 
type 'open' or 'close' when prompted.
'''

import json
from pyrobot import Robot

robot = Robot('locobot')

# Move to start position.
robot.arm.set_ee_pose_pitch_roll([0.418, 0., 0.3], pitch=1.57, roll=0, plan=True)

def open_gripper():
    robot.gripper.open()

def close_gripper():
    robot.gripper.close()

def move(position):
    robot.arm.set_ee_pose_pitch_roll(position, pitch=1.57, roll=0, plan=True)

while True:
    command = input("Command (open, close, move, exit): ")
    if command.lower() == 'open':
        open_gripper()
    elif command.lower() == 'close':
        close_gripper()
    elif command.lower() == 'move':
        # move to absolute position
        position = input('Enter a list [x,y,z]: ')
        # change string to python list
        position = json.loads(position)
        move(position)
    elif command.lower() == 'exit':
        robot.arm.set_ee_pose_pitch_roll([0.418, 0., 0.3], pitch=1.57, roll=0, plan=True)
        break
    else:
        print('please enter a valid command.')


# TODO: add GUI for convenience? 