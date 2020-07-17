# Script to visualize the objective values obtained by each initial inference made by GP regression during training

import numpy as np
import matplotlib.pyplot as plt
import learn_ProMP_from_demonstrations as lpd
import compute_min_s_ED as cm

np.random.seed(2)


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


# Computes the distances between the via point and points on the sample trajectories
def compute_dist(traj_samples, viapoint, rng=None):  # rng is a variable that determines if it's distance to start, end or other distance.
    n_samples = traj_samples.shape[0]
    dofs = viapoint.shape[0]
    max_length = int(traj_samples.shape[1] / dofs)

    # Center
    if rng is None:
        all_center_dists = np.zeros((n_samples, max_length))

        for t in range(max_length):
            center_dist = traj_samples[:, [t + d * max_length for d in range(dofs)]] - viapoint
            center_dist = np.sqrt(np.sum(center_dist ** 2, axis=1))
            all_center_dists[:, t] = center_dist

        min_distance_index = np.argmin(all_center_dists, axis=1)  # Find the index (time step) of the closest point along the trajectory to the via point.
                                                                  # This line computes an index (time step) for each sampled trajectory. The output has shape (n_samples, 1)
                                                                  # QUESTION: Wouldn't it be more efficient to use min instead or argmin?
        important_points = np.zeros((n_samples, dofs))
        for s in range(n_samples):
            t = min_distance_index[s]
            important_points[s, :] = traj_samples[s, [t + d * max_length for d in range(dofs)]]

        dist = important_points - viapoint
        dist = np.sqrt(np.sum(dist ** 2, axis=1))

    # Start or end
    else:
        dist = traj_samples[:, rng] - viapoint
        dist = np.sqrt(np.sum(dist ** 2, axis=1))
    return dist.reshape(len(dist), 1)  # (n_samples, 1)


# Computes the minimum signed Euclidean distance between each sampled trajectory and the obstacles
def compute_min_s_ED_trajectories(traj_samples, wall_corners):
    n_samples = traj_samples.shape[0]
    dofs = 2
    max_length = int(traj_samples.shape[1] / dofs)

    min_s_ED_trajectories = np.zeros((n_samples, 1))

    for i in range(n_samples):
        traj = traj_samples[i, :].reshape(1, dofs*max_length)

        time_range = range(0, max_length, 1)
        min_s_ED_trajectory = np.zeros((len(time_range), 1))

        for j in range(len(time_range)):
            t = time_range[j]
            point = traj[0, [t, t+max_length]].reshape(2,1)
            min_s_ED_trajectory[j] = cm.compute_min_s_ED(point, wall_corners)

        # Select the minimum signed Euclidean distance for the whole trajectory
        min_s_ED_trajectories[i] = min(min_s_ED_trajectory)

    return min_s_ED_trajectories


n_samples = 100
n_objectives = 3
n_inferences = 1000
num_gaba = 10
dofs = 2
max_length = 200

n_demonstrated_envs = 30
n_demonstrations_per_env = 3

n_demonstrations = n_demonstrated_envs*n_demonstrations_per_env

promp_from_demonstration = lpd.learn_promp(num_gaba, n_demonstrations)

mean_objectives_r = np.zeros((n_objectives, n_inferences))
std_objectives_r = np.zeros((n_objectives, n_inferences))
mean_objectives = np.zeros((n_objectives, n_inferences))
std_objectives = np.zeros((n_objectives, n_inferences))

start = [n * max_length for n in range(dofs)]  # Indices for the values of DoFs at the start of a trajectory
end = [n * max_length - 1 for n in range(1, dofs + 1)]  # Indices for the values of DoFs at the end of a trajectory

for i in range(n_inferences):

    print(i)

#################################################################################################

    ms_r = np.load('GP_inferences/ms_{}.npy'.format(i))
    Ds_r = np.load('GP_inferences/Ds_{}.npy'.format(i))
    env = np.load('GP_inferences/env_{}.npy'.format(i))

    viapoints = np.array([env[0:2], env[4:6]])  # [start, end]
    wall_corners = compute_wall_corners(env[2], env[3])

    weight_samples_r, traj_samples_r = sample_trajectories(n_samples, ms_r, Ds_r, num_gaba, promp_from_demonstration.block_PSI)

    objective_0_val_r = compute_dist(traj_samples_r, viapoints[0], start)
    objective_0_mean_r = np.mean(objective_0_val_r)
    objective_0_std_r = np.std(objective_0_val_r)
    mean_objectives_r[0, i] = objective_0_mean_r
    std_objectives_r[0, i] = objective_0_std_r

    objective_1_val_r = compute_min_s_ED_trajectories(traj_samples_r, wall_corners)
    objective_1_mean_r = np.mean(objective_1_val_r)
    objective_1_std_r = np.std(objective_1_val_r)
    mean_objectives_r[1, i] = objective_1_mean_r
    std_objectives_r[1, i] = objective_1_std_r

    objective_2_val_r = compute_dist(traj_samples_r, viapoints[1], end)
    objective_2_mean_r = np.mean(objective_2_val_r)
    objective_2_std_r = np.std(objective_2_val_r)
    mean_objectives_r[2, i] = objective_2_mean_r
    std_objectives_r[2, i] = objective_2_std_r

###################################################################################################

    ms = np.load('GP_inferences_only_based_on_demonstrations/ms_{}.npy'.format(i))
    Ds = np.load('GP_inferences_only_based_on_demonstrations/Ds_{}.npy'.format(i))
    env = np.load('GP_inferences_only_based_on_demonstrations/env_{}.npy'.format(i))

    viapoints = np.array([env[0:2], env[4:6]])  # [start, end]
    wall_corners = compute_wall_corners(env[2], env[3])

    weight_samples, traj_samples = sample_trajectories(n_samples, ms, Ds, num_gaba,
                                                           promp_from_demonstration.block_PSI)

    objective_0_val = compute_dist(traj_samples, viapoints[0], start)
    objective_0_mean = np.mean(objective_0_val)
    objective_0_std = np.std(objective_0_val)
    mean_objectives[0, i] = objective_0_mean
    std_objectives[0, i] = objective_0_std

    objective_1_val = compute_min_s_ED_trajectories(traj_samples, wall_corners)
    objective_1_mean = np.mean(objective_1_val)
    objective_1_std = np.std(objective_1_val)
    mean_objectives[1, i] = objective_1_mean
    std_objectives[1, i] = objective_1_std

    objective_2_val = compute_dist(traj_samples, viapoints[1], end)
    objective_2_mean = np.mean(objective_2_val)
    objective_2_std = np.std(objective_2_val)
    mean_objectives[2, i] = objective_2_mean
    std_objectives[2, i] = objective_2_std

###################################################################################################

fig = plt.figure()
ax0 = fig.add_subplot(131)
ax0.grid(True)
ax0.fill_between(range(n_inferences), mean_objectives[0, :] - 2.0 * std_objectives[0, :], mean_objectives[0, :] + 2.0 * std_objectives[0, :],
                         alpha=0.5, color='b')
ax0.plot(range(n_inferences), mean_objectives[0, :], color='b', label='without PRO')
ax0.fill_between(range(n_inferences), mean_objectives_r[0, :] - 2.0 * std_objectives_r[0, :], mean_objectives_r[0, :] + 2.0 * std_objectives_r[0, :],
                         alpha=0.5, color='r')
ax0.plot(range(n_inferences), mean_objectives_r[0, :], color='r', label='with PRO')
ax0.set(xlabel='inference', ylabel='start distance $(\mu \pm 2\sigma)$')
ax0.legend()

ax1 = fig.add_subplot(132)
ax1.grid(True)
ax1.fill_between(range(n_inferences), mean_objectives[1, :] - 2.0 * std_objectives[1, :], mean_objectives[1, :] + 2.0 * std_objectives[1, :],
                         alpha=0.5, color='b')
ax1.plot(range(n_inferences), mean_objectives[1, :], color='b', label='without PRO')
ax1.fill_between(range(n_inferences), mean_objectives_r[1, :] - 2.0 * std_objectives_r[1, :], mean_objectives_r[1, :] + 2.0 * std_objectives_r[1, :],
                         alpha=0.5, color='r')
ax1.plot(range(n_inferences), mean_objectives_r[1, :], color='r', label='with PRO')
ax1.set(xlabel='inference', ylabel='min. signed Euclidean distance $(\mu \pm 2\sigma)$')
ax1.legend()

ax2 = fig.add_subplot(133)
ax2.grid(True)
ax2.fill_between(range(n_inferences), mean_objectives[2, :] - 2.0 * std_objectives[2, :], mean_objectives[2, :] + 2.0 * std_objectives[2, :],
                         alpha=0.5, color='b')
ax2.plot(range(n_inferences), mean_objectives[2, :], color='b', label='without PRO')
ax2.fill_between(range(n_inferences), mean_objectives_r[2, :] - 2.0 * std_objectives_r[2, :], mean_objectives_r[2, :] + 2.0 * std_objectives_r[2, :],
                         alpha=0.5, color='r')
ax2.plot(range(n_inferences), mean_objectives_r[2, :], color='r', label='with PRO')
ax2.set(xlabel='inference', ylabel='end distance $(\mu \pm 2\sigma)$')
ax2.legend()

plt.show()







