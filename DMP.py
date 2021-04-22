'''
Fits a DMP from EE xyz position velocity and acceleration. 
Generates trajectory for the learned skill.
'''

import math
import numpy as np
import matplotlib.pyplot as plt

from pyrobot import Robot

class DMP:

    def __init__(self, num_basis):
        self.K = 25 * 25 / 4
        self.B = 25
        self.centers = np.linspace(0, 1, num_basis) # Basis function centers
        self.width = (0.65 * (1. / (num_basis - 1.)) ** 2) # Basis function widths
        self.duration = 3 # TODO: Load this from recording
        self.dt = 0.003

    def learn_weights(self, filename):
        '''
        Learn DMP weights from recordings
        '''
        data = np.load(filename)
        times = data['time']
        positions = data['pos']
        velocities = data['vel']
        accelerations = data['acc']
        duration = 3.0 # TODO: Load this from recording

        PHI = []
        forcing_function = []
        for i, time in enumerate(times):
            phi = [math.exp(-0.5 * ((time / duration) - center) ** 2 / self.width) for center in self.centers]
            phi = phi / np.sum(phi)
            PHI.append(phi)

            f = ((accelerations[i] * duration ** 2) - self.K * (positions[-1] - positions[i]) + self.B * (velocities[i] * duration)) / (positions[-1] - positions[0])
            forcing_function.append(f)

        # Calculate weights via linear regression
        self.weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(PHI),PHI)),np.transpose(PHI)), forcing_function)

        # Save the weights file to save relearning time
        np.save('DMPweights.npy', self.weights)

    def load_weights(self, filename):
        '''
        Load weights from npy file
        '''
        self.weights = np.load(filename)

    def generate_traj(self, start, goal):
        '''
        From learned weights, generate a trajectory from start to goal.
        start, goal: python list of floats xyz positions
        '''
        pose_start = np.array(start)
        pose = np.copy(pose_start)
        pose_goal = np.array(goal)
        vel = np.zeros(3)
        acc = np.zeros(3)
        dmp_trajectory = []
        dmp_trajectory.append(pose_start.tolist())

        t = 0
        for i in range(1500):
            t = t + self.dt
            if t < self.duration:
                Phi = [math.exp(-0.5 * ((t/self.duration) - center) ** 2 / self.width) for center in self.centers]
                Phi = Phi / np.sum(Phi)
                force = np.dot(Phi, self.weights)
            else: 
                force = 0
            acc = self.K * (pose_goal - pose) / (self.duration ** 2) - self.B * vel / self.duration + (pose_goal - pose_start) * force / (self.duration ** 2)
            vel = vel + acc * self.dt
            # import pdb;pdb.set_trace()
            pose = pose + vel * self.dt
            dmp_trajectory.append(pose.tolist())

        return dmp_trajectory


if __name__ == "__main__":
    DMP = DMP(5)
    print('learning weights')
    DMP.learn_weights('right.npz')
    # DMP.load_weights('DMPweights.npy')
    print('generating trajectory')
    trajectory = DMP.generate_traj([0.338, 0.1, 0.26], [0.338, -0., 0.26])
    # import pdb; pdb.set_trace()
    fig,axes = plt.subplots(3)
    axes[0].plot([trajectory[i][0] for i in range(len(trajectory))])
    axes[0].annotate(str(trajectory[0][0]),xy=(0,trajectory[0][0]))
    axes[0].annotate(str(trajectory[-1][0]),xy=(1000,trajectory[-1][0]))
    axes[0].set_ylabel('X(m)')
    # print(dir(axes[0].yaxis))s


    axes[1].plot([trajectory[i][1] for i in range(len(trajectory))])
    axes[1].set_ylabel('Y(m)')
    axes[2].plot([trajectory[i][2] for i in range(len(trajectory))])
    axes[2].set_ylabel('Z(m)')

    # plt.show()

    robot = Robot('locobot')
    print('moving')
    # TODO: set position directly in generate_traj?
    for position in trajectory:
        robot.arm.set_ee_pose_pitch_roll(position, pitch=1.57, roll=0, plan=False)
        # time.sleep(0.1)
