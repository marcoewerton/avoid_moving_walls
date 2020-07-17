# This script assumes a previously trained model
# The ProMP adapts to environments changing on the fly


########### Import libraries ########################################################

import numpy as np
import matplotlib.pyplot as plt
import learn_ProMP_from_demonstrations as lpd
from numpy.linalg import inv
import rwpo
import timeit
import time
import PD_controller as pd
import plant
from scipy.interpolate import interp1d

# tic = timeit.default_timer()

########### Load Learned Model ######################################################

learned_model_folder = "Learned_Model"
known_envs = np.load(learned_model_folder + '/known_envs.npy')
known_mean_weights = np.load(learned_model_folder + '/known_mean_weights.npy')
known_var_weights = np.load(learned_model_folder + '/known_var_weights.npy')
mu = np.load(learned_model_folder + '/mu.npy')
noise_var = np.load(learned_model_folder + '/noise_var.npy')
max_length = np.load(learned_model_folder + '/max_length.npy')
block_PSI = np.load(learned_model_folder + '/block_PSI.npy')
num_gaba = np.load(learned_model_folder + '/num_gaba.npy')


########### Set random seed #########################################################

np.random.seed(10)


########### Functions ###############################################################

# Kernel function
def kernel(x, y):
    return np.exp(-0.001*(x-y).T @ (x-y))  # squared exponential


# Function to sample a new random environment
def sample_new_env():
    wx = 2 + np.random.rand() * 6
    wy = 1 + np.random.rand() * 8

    #start_x = np.random.rand() * (wx - 1.5)
    #start_y = np.random.rand() * 10
    start_x = 0
    start_y = 5

    end_x = np.random.rand() * (10 - (wx + 1.5)) + wx + 1.5
    end_y = np.random.rand() * 10

    new_rand_env = np.array([start_x, start_y, wx, wy, end_x, end_y])
    return new_rand_env


# Function to slightly change an environment
def change_env(environment, dsy, dwx, dwy, dey):
    # start_x = environment[0]
    # start_y = environment[1]
    start_x = 0
    start_y = 5
    wx = environment[2]
    wy = environment[3]
    end_x = environment[4]
    end_y = environment[5]

    wx = wx + dwx
    #start_x = start_x + dwx
    end_x = end_x + dwx
    #if start_x < 0 or start_x > wx - 1.5:
    #    start_x = start_x - dwx
    if end_x < wx + 1.5 or end_x > 10:
        end_x = end_x - dwx
    if wx < 2 or wx > 8:
        wx = wx - dwx
        dwx = -dwx

    wy = wy + dwy
    if wy < 1 or wy > 9:
        wy = wy - dwy
        dwy = -dwy

    #start_y =  start_y + dsy
    #if start_y < 0 or start_y > 10:
    #    start_y = start_y - dsy
    #    dsy = -dsy

    end_y = end_y + dey
    if end_y < 0 or end_y > 10:
        end_y = end_y - dey
        dey = -dey

    changed_env = np.array([start_x, start_y, wx, wy, end_x, end_y])
    return changed_env, dsy, dwx, dwy, dey


# Functions to compute the covariance matrix in the form Caa, Cab, Cba, Cbb
def compute_cov_matrix1(known_environments):
    _Cbb = np.zeros((known_environments.shape[1], known_environments.shape[1]))
    for index1 in range(known_environments.shape[1]):
        _Cbb[index1, index1] = kernel(known_environments[:, index1], known_environments[:, index1])
        for index2 in range(index1 + 1, known_environments.shape[1]):
            _Cbb[index1, index2] = _Cbb[index2, index1] = kernel(known_environments[:, index1], known_environments[:, index2])
    return _Cbb

def compute_cov_matrix2(new_environment, known_environments):
    _Caa = np.array(kernel(new_environment, new_environment)).reshape((1, 1))
    _Cab = np.zeros((1, known_environments.shape[1]))
    for known_env_index in range(known_environments.shape[1]):
        _Cab[0, known_env_index] = kernel(new_environment, known_environments[:, known_env_index])
    _Cba = _Cab.T
    return _Caa, _Cab, _Cba


# Function to plot an environment for the first time
def plot_environment(environment):

    start = np.array([environment[0], environment[1]])
    wall_corners = compute_wall_corners(environment[2], environment[3])
    end = np.array([environment[4], environment[5]])

    line_start, = plt.plot(start[0], start[1], 'gx', markersize=12)
    line_wall_1, = plt.plot([wall_corners[0][0], wall_corners[1][0], wall_corners[2][0], wall_corners[3][0], wall_corners[0][0]],
              [wall_corners[0][1], wall_corners[1][1], wall_corners[2][1], wall_corners[3][1], wall_corners[0][1]], 'b',
              linewidth=2)
    line_wall_2, = plt.plot([wall_corners[4][0], wall_corners[5][0], wall_corners[6][0], wall_corners[7][0], wall_corners[4][0]],
              [wall_corners[4][1], wall_corners[5][1], wall_corners[6][1], wall_corners[7][1], wall_corners[4][1]], 'b',
              linewidth=2)
    line_end, = plt.plot(end[0], end[1], 'rx', markersize=12)

    return line_start, line_wall_1, line_wall_2, line_end


# Function to update environment
def update_environment(line_start, line_wall_1, line_wall_2, line_end, environment):

    start = np.array([environment[0], environment[1]])
    wall_corners = compute_wall_corners(environment[2], environment[3])
    end = np.array([environment[4], environment[5]])

    line_start.set_data(start[0], start[1])
    line_wall_1.set_data([wall_corners[0][0], wall_corners[1][0], wall_corners[2][0], wall_corners[3][0], wall_corners[0][0]],
              [wall_corners[0][1], wall_corners[1][1], wall_corners[2][1], wall_corners[3][1], wall_corners[0][1]])
    line_wall_2.set_data([wall_corners[4][0], wall_corners[5][0], wall_corners[6][0], wall_corners[7][0], wall_corners[4][0]],
              [wall_corners[4][1], wall_corners[5][1], wall_corners[6][1], wall_corners[7][1], wall_corners[4][1]])
    line_end.set_data(end[0], end[1])


# Function to compute wall corners
def compute_wall_corners(wx, wy):
    _wall_corners = [np.array([[wx - 1], [10]]), np.array([[wx - 1], [wy + 1]]), np.array([[wx + 1], [wy + 1]]),
                    np.array([[wx + 1], [10]]), np.array([[wx - 1], [wy - 1]]), np.array([[wx - 1], [0]]),
                    np.array([[wx + 1], [0]]), np.array([[wx + 1], [wy - 1]])]
    return _wall_corners


# Function to sample trajectories
def sample_trajectories(n_samples, mu_w, var_w, n_gbf, block_PSI):
    _weight_samples = np.random.multivariate_normal(mu_w.reshape((mu_w.size,)),
                                                   np.diag(var_w.reshape(2*n_gbf, )), n_samples)
    # shape = (n_samples, dofs*max_length)
    _traj_samples = np.dot(_weight_samples, np.transpose(block_PSI))
    return _weight_samples, _traj_samples


# Functions related to recording new trajectories

pressed = False
trajectory_x, trajectory_y = [], []


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

    pressed = False
    trajectory = np.array([trajectory_x, trajectory_y])

    file_name = "./New_demonstrations/traj"+str(traj_count)+".out"
    np.savetxt(file_name, trajectory.T, delimiter=',')

    traj_count += 1

    # Reset current trajectory
    trajectory_x, trajectory_y = [], []


########### Main code #############################################################

n_samples = 10
n_iter = 200  # Maximum number of iterations. RWPO may reach convergence before that.
n_t_steps = 200  # Number of time steps.
total_t_steps = 250  # Total time (including time after end position has been reached).
dt = 0.01

# Precomputation
new_mean_weights = np.zeros((2*num_gaba, 1))
new_var_weights = np.zeros((2*num_gaba, 1))
Cbb = compute_cov_matrix1(known_envs)
inv_Cbb_plus_noise_list = []
yb_list = []
for w_index in range(2*num_gaba):
    w_var_matrix = np.diag(known_var_weights[w_index, :])
    inv_Cbb_plus_noise_list.append(inv(Cbb + w_var_matrix))
    yb = known_mean_weights[w_index, :].reshape(known_envs.shape[1], 1)
    yb_list.append(yb)

# for each new random environment, infer w with mean and standard deviation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
# sample a new random environment
new_env = sample_new_env()
dsy, dwx, dwy, dey = [0.1, 0.1, 0.1, 0.1]

traj_lines = []
point = np.array([new_env[0], new_env[1], 0, 0])

#plt.pause(20)

for i in range(total_t_steps):
    if i < n_t_steps:
        print('New environment {}'.format(i))

        # change environment
        new_env, dsy, dwx, dwy, dey = change_env(new_env, dsy, dwx, dwy, dey)

    if i == 0:
        line_start, line_wall_1, line_wall_2, line_end = plot_environment(new_env)
    else:
        update_environment(line_start, line_wall_1, line_wall_2, line_end, new_env)

    for w_index in range(2*num_gaba):  # There is one GP for each weight
        Caa, Cab, Cba = compute_cov_matrix2(new_env, known_envs)
        yb = yb_list[w_index]
        # Inference
        m = mu[w_index] + Cab @ inv_Cbb_plus_noise_list[w_index] @ (yb_list[w_index] - mu[w_index]) # mu_a = mu_b = mu[w_index]
        D = Caa + noise_var[w_index] - Cab @ inv_Cbb_plus_noise_list[w_index] @ Cba
        new_mean_weights[w_index] = m
        new_var_weights[w_index] = D

    viapoints = np.array([new_env[0:2], new_env[4:6]])  # [start, end]
    wall_corners = compute_wall_corners(new_env[2], new_env[3])

    # # Save GP inference
    # np.save('GP_inferences2/m_{}'.format(i), m)
    # np.save('GP_inferences2/D_{}'.format(i), D)
    # np.save('GP_inferences2/env_{}'.format(i), new_env)

   # Visualize GP inference

    # sample trajectories from the posterior computed with GP regression
    weight_samples, traj_samples = sample_trajectories(n_samples, new_mean_weights, new_var_weights, num_gaba, block_PSI)
    # plot sampled trajectories
    for sample_index in range(n_samples):
        if i == 0:
            traj_line, = plt.plot(traj_samples[sample_index, 0:max_length], traj_samples[sample_index, max_length:2*max_length], color='0.7')
            traj_lines.append(traj_line)
        else:
            traj_lines[sample_index].set_data(traj_samples[sample_index, 0:max_length], traj_samples[sample_index, max_length:2*max_length])

    # Compute mean trajectory from the GP inference and visualize it
    mean_trajectory = np.dot(new_mean_weights.T, np.transpose(block_PSI)).reshape(-1, )
    old_x = np.linspace(0, 1, np.int(mean_trajectory.size/2))
    new_x = np.linspace(0, 1, n_t_steps)
    interp = interp1d(old_x, mean_trajectory.reshape((2, max_length), order='C'))
    mean_trajectory = interp(new_x)
    if i == 0:
        mean_traj_line, = plt.plot(mean_trajectory[0, :], mean_trajectory[1, :], color='r')
    else:
        mean_traj_line.set_data(mean_trajectory[0, :], mean_trajectory[1, :])

    mean_velocity = np.diff(mean_trajectory)/dt
    mean_velocity = np.hstack([mean_velocity, np.array([[0], [0]])])

    if i < n_t_steps:
        # Determine current goal and plot it
        current_goal = np.array([mean_trajectory[0, i], mean_trajectory[1, i], mean_velocity[0, i], mean_velocity[1, i]])
        if i == 0:
            current_goal_line, = plt.plot(current_goal[0], current_goal[1], 'k*')
        else:
            current_goal_line.set_data(current_goal[0], current_goal[1])

    # Move the point particle and plot it
    acceleration = pd.PD_controller(point, current_goal)
    point = plant.plant(acceleration, point, dt)

    if i == 0:
        point_line, = plt.plot(point[0], point[1], 'mo')
    else:
        point_line.set_data(point[0], point[1])


    # plt.ion()
    # plt.show()
    plt.pause(dt)
    # plt.close(fig)


    # # Optimize inference using RWPO
    # learnt_pol, converged = rwpo.optimize(viapoints, n_iter, n_samples, max_length, num_gaba, block_PSI, wall_corners, new_mean_weights, new_var_weights, sample_trajectories)
    # new_mean_weights = learnt_pol.mu_w
    # new_var_weights = learnt_pol.var_w
    #
    # # # sample trajectories from the optimized distribution
    # # weight_samples, traj_samples = sample_trajectories(n_samples, new_mean_weights, new_var_weights, num_gaba, block_PSI)
    #
    # # plot new environment
    # plot_environment(new_env)
    #
    # # plot sampled trajectories
    # for sample_index in range(n_samples):
    #     plt.plot(traj_samples[sample_index, 0:max_length], traj_samples[sample_index, max_length:2*max_length], color='0.7')
    #
    # plt.show()

    # user_answer = input('Is this solution acceptable? [y or n]')
    # if user_answer is 'n':
    #     traj_count = 0
    #     print('Please provide 3 demonstrations.')
    #     for demonstration_index in range(3):
    #         fig, ax = plot_environment(new_env)
    #         line, = ax.plot([0], [0])
    #
    #         cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
    #         cid2 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    #         cid3 = fig.canvas.mpl_connect('button_release_event', on_release)
    #
    #         plt.show()
    #
    #     promp_from_new_demonstrations = lpd.learn_promp(num_gaba, 3, folder="New_demonstrations")
    #     new_mean_weights = np.mean(promp_from_new_demonstrations.weight_matrix, axis=1).reshape(-1, 1)
    #     new_var_weights = np.var(promp_from_new_demonstrations.weight_matrix, axis=1).reshape(-1, 1)

    # if converged:
    #     known_envs = np.hstack((known_envs, new_env.reshape((6, 1))))
    #     known_mean_weights = np.hstack((known_mean_weights, new_mean_weights))
    #     known_var_weights = np.hstack((known_var_weights, new_var_weights))
    #
    #     noise_var = np.mean(known_var_weights, axis=1)
    #     mu = np.mean(known_var_weights, axis=1)
    #
    #     # Save Learned Model
    #     learned_model_folder = "Learned_Model"
    #     np.save(learned_model_folder + '/known_envs', known_envs)
    #     np.save(learned_model_folder + '/known_mean_weights', known_mean_weights)
    #     np.save(learned_model_folder + '/known_var_weights', known_var_weights)
    #     np.save(learned_model_folder + '/mu', mu)
    #     np.save(learned_model_folder + '/noise_var', noise_var)
    #     np.save(learned_model_folder + '/max_length', max_length)
    #     np.save(learned_model_folder + '/block_PSI', block_PSI)
    #     np.save(learned_model_folder + '/num_gaba', num_gaba)
    #
    #     print("Model has been updated.")
    # else:
    #     print("Model has not been updated.")

# toc = timeit.default_timer()
# print(toc - tic)
