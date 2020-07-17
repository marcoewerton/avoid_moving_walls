########### Import libraries ########################################################

import numpy as np
import matplotlib.pyplot as plt
import learn_ProMP_from_demonstrations as lpd
from numpy.linalg import inv
import rwpo
import timeit

tic = timeit.default_timer()

########### Set random seed #########################################################

np.random.seed(2)


########### Set number of basis functions of the ProMPs #############################

num_gaba = 10  # Number of Gaussian basis functions

n_demonstrated_envs = 30
n_demonstrations_per_env = 3

n_demonstrations = n_demonstrated_envs*n_demonstrations_per_env


########### Determine ProMP based on the demonstrations #############################

promp_from_demonstration = lpd.learn_promp(num_gaba, n_demonstrations)

known_mean_weights = np.zeros((2*num_gaba, n_demonstrated_envs))
for i in range(n_demonstrated_envs):
    known_mean_weights[:, i] = np.mean(promp_from_demonstration.weight_matrix[:, n_demonstrations_per_env*i:n_demonstrations_per_env*i + n_demonstrations_per_env], axis=1)

known_var_weights = np.zeros((2*num_gaba, n_demonstrated_envs))
for i in range(n_demonstrated_envs):
    known_var_weights[:, i] = np.var(promp_from_demonstration.weight_matrix[:, n_demonstrations_per_env*i:n_demonstrations_per_env*i + n_demonstrations_per_env], axis=1)

noise_var = np.mean(known_var_weights, axis=1)  # average variance for each weight based on the demonstrations

mu = np.mean(known_mean_weights, axis=1)  # average value for each weight based on the demonstrations

########### Functions ###############################################################

# Kernel function
def kernel(x, y):
    return np.exp(-0.001*(x-y).T @ (x-y))  # squared exponential


# Function to sample a new random environment
def sample_new_env():
    wx = 2 + np.random.rand() * 6
    wy = 1 + np.random.rand() * 8

    start_x = np.random.rand() * (wx - 1.5)
    start_y = np.random.rand() * 10

    end_x = np.random.rand() * (10 - (wx + 1.5)) + wx + 1.5
    end_y = np.random.rand() * 10

    new_rand_env = np.array([start_x, start_y, wx, wy, end_x, end_y])
    return new_rand_env


# Function to compute the covariance matrix in the form Caa, Cab, Cba, Cbb
def compute_cov_matrix(new_environment, known_environments):
    _Caa = np.array(kernel(new_environment, new_environment)).reshape((1, 1))
    _Cab = np.zeros((1, known_environments.shape[1]))
    for known_env_index in range(known_environments.shape[1]):
        _Cab[0, known_env_index] = kernel(new_environment, known_environments[:, known_env_index])
    _Cba = _Cab.T
    _Cbb = np.zeros((known_environments.shape[1], known_environments.shape[1]))
    for index1 in range(known_environments.shape[1]):
        _Cbb[index1, index1] = kernel(known_environments[:, index1], known_environments[:, index1])
        for index2 in range(index1 + 1, known_environments.shape[1]):
            _Cbb[index1, index2] = _Cbb[index2, index1] = kernel(known_environments[:, index1], known_environments[:, index2])
    return _Caa, _Cab, _Cba, _Cbb


# Function to plot an environment
def plot_environment(environment):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    start = np.array([environment[0], environment[1]])
    wall_corners = compute_wall_corners(environment[2], environment[3])
    end = np.array([environment[4], environment[5]])

    plt.plot(start[0], start[1], 'gx', markersize=12)
    plt.plot([wall_corners[0][0], wall_corners[1][0], wall_corners[2][0], wall_corners[3][0], wall_corners[0][0]],
              [wall_corners[0][1], wall_corners[1][1], wall_corners[2][1], wall_corners[3][1], wall_corners[0][1]], 'b',
              linewidth=2)
    plt.plot([wall_corners[4][0], wall_corners[5][0], wall_corners[6][0], wall_corners[7][0], wall_corners[4][0]],
              [wall_corners[4][1], wall_corners[5][1], wall_corners[6][1], wall_corners[7][1], wall_corners[4][1]], 'b',
              linewidth=2)
    plt.plot(end[0], end[1], 'rx', markersize=12)

    return fig, ax


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

n_samples = 100
n_iter = 200  # Maximum number of iterations. RWPO may reach convergence before that.

# load environments from the demonstrations
known_envs = None
for i in range(n_demonstrated_envs):
    string = "Demonstrations/" + "env" + str(n_demonstrations_per_env*i) + ".out"
    env = np.genfromtxt(string, delimiter=',')
    if known_envs is None:
        known_envs = env.reshape((6, 1))
    else:
        known_envs = np.hstack((known_envs, env.reshape((6, 1))))

# for each new random environment, infer w with mean and standard deviation
for i in range(1000):
    print('New environment {}'.format(i))

    # # sample a new random environment
    # new_env = sample_new_env()
    # load an environment used by the main script
    new_env = np.load('GP_inferences/env_{}.npy'.format(i))

    new_mean_weights = np.zeros((2*num_gaba, 1))
    new_var_weights = np.zeros((2*num_gaba, 1))
    for w_index in range(2*num_gaba):  # There is one GP for each weight
        Caa, Cab, Cba, Cbb = compute_cov_matrix(new_env, known_envs)
        yb = known_mean_weights[w_index, :].reshape(known_envs.shape[1], 1)
        w_var_matrix = np.diag(known_var_weights[w_index, :])
        # Inference
        m = mu[w_index] + Cab @ inv(Cbb + w_var_matrix) @ (yb - mu[w_index]) # mu_a = mu_b = mu[w_index]
        D = Caa + noise_var[w_index] - Cab @ inv(Cbb + w_var_matrix) @ Cba
        new_mean_weights[w_index] = m
        new_var_weights[w_index] = D


    viapoints = np.array([new_env[0:2], new_env[4:6]])  # [start, end]
    wall_corners = compute_wall_corners(new_env[2], new_env[3])

    # Save GP inference
    np.save('GP_inferences_only_based_on_demonstrations/ms_{}'.format(i), new_mean_weights)
    np.save('GP_inferences_only_based_on_demonstrations/Ds_{}'.format(i), new_var_weights)
    np.save('GP_inferences_only_based_on_demonstrations/env_{}'.format(i), new_env)


    # # Visualize GP inference
    #
    # # plot new environment
    # plot_environment(new_env)
    # # sample trajectories from the posterior computed with GP regression
    # weight_samples, traj_samples = sample_trajectories(n_samples, new_mean_weights, new_var_weights, num_gaba,
    #                                                    promp_from_demonstration.block_PSI)
    # # plot sampled trajectories
    # for sample_index in range(n_samples):
    #     plt.plot(traj_samples[sample_index, 0:promp_from_demonstration.max_length], traj_samples[sample_index,
    #                                                                                 promp_from_demonstration.max_length:2 * promp_from_demonstration.max_length],
    #              color='0.7')
    #
    # plt.show()




toc = timeit.default_timer()
print(toc - tic)
