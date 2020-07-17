# for each environment, record a number of demonstrations

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

pressed = False
trajectory_x, trajectory_y = [], []
traj_count = 0


def on_press(event):
    global pressed
    pressed = True


def on_move(event):
    global pressed
    global trajectory_x
    global trajectory_y
    global line
    if pressed:
        trajectory_x.append(event.xdata)
        trajectory_y.append(event.ydata)
        line.set_data(trajectory_x, trajectory_y)
        line.figure.canvas.draw()


def on_release(event):
    global pressed
    global trajectory_x
    global trajectory_y
    global traj_count
    global window
    global start
    global end

    pressed = False
    trajectory = np.array([trajectory_x, trajectory_y])
    print("Start:", trajectory_x[0], trajectory_y[0])
    print("End:", trajectory_x[len(trajectory_x)-1], trajectory_y[len(trajectory_y)-1])

    # Save a number of drawn trajectories and quit
    file_name = "./Demonstrations/traj"+str(traj_count)+".out"
    np.savetxt(file_name, trajectory.T, delimiter=',')
    file_name = "./Demonstrations/env" + str(traj_count) + ".out"
    np.savetxt(file_name, np.hstack((start, window, end)), delimiter=',')

    traj_count += 1

    # Reset current trajectory
    trajectory_x, trajectory_y = [], []


n_demonstrated_envs = 30
n_demonstrations_per_env = 3
for i in range(n_demonstrated_envs):

    wx = 2 + np.random.rand() * 6
    wy = 1 + np.random.rand() * 8
    window = np.array([wx, wy])
    wall_corners = [np.array([[wx-1], [10]]), np.array([[wx-1], [wy+1]]), np.array([[wx+1], [wy+1]]), np.array([[wx+1], [10]]), np.array([[wx-1], [wy-1]]), np.array([[wx-1], [0]]), np.array([[wx+1], [0]]), np.array([[wx+1], [wy-1]])]

    start_x = np.random.rand() * (wx-1.5)
    start_y = np.random.rand() * 10
    start = np.array([start_x, start_y])

    end_x = np.random.rand()*(10 - (wx+1.5)) + wx + 1.5
    end_y = np.random.rand()*10
    end = np.array([end_x, end_y])

    for j in range(n_demonstrations_per_env):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True)

        plt.plot(start[0], start[1], 'gx', markersize=12)
        plt.plot([wall_corners[0][0], wall_corners[1][0], wall_corners[2][0], wall_corners[3][0], wall_corners[0][0]], [wall_corners[0][1], wall_corners[1][1], wall_corners[2][1], wall_corners[3][1], wall_corners[0][1]], 'b', linewidth=2)
        plt.plot([wall_corners[4][0], wall_corners[5][0], wall_corners[6][0], wall_corners[7][0], wall_corners[4][0]], [wall_corners[4][1], wall_corners[5][1], wall_corners[6][1], wall_corners[7][1], wall_corners[4][1]], 'b', linewidth=2)
        plt.plot(end[0], end[1], 'rx', markersize=12)

        line, = ax.plot([0], [0])

        cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
        cid2 = fig.canvas.mpl_connect('motion_notify_event', on_move)
        cid3 = fig.canvas.mpl_connect('button_release_event', on_release)

        plt.show()