### RWPO ###
import numpy as np
import compute_min_s_ED as cm


# Represents a learnt optimal policy
class Learnt_pol:

    def __init__(self, all_objective_vals, trajs, relevance, mu_w, var_w):
        self.all_objective_vals = all_objective_vals
        self.trajs = trajs
        self.relevance = relevance
        self.mu_w = mu_w
        self.var_w = var_w


# Optimizes distribution of trajectories using RWPO
# viapoints here has only the start and the end via points.
def optimize(viapoints, n_iter, n_samples, max_length, num_gaba, block_PSI, wall_corners, mu_w, var_w, sample_trajectories, beta=20):
    n_relfun = len(viapoints) + 1  # Number of relevance functions. There is one for each objective.
    dofs = viapoints.shape[1]  # Number of degrees of freedom

    start = [n * max_length for n in range(dofs)]  # Indices for the values of DoFs at the start of a trajectory
    end = [n * max_length - 1 for n in range(1, dofs + 1)]  # Indices for the values of DoFs at the end of a trajectory

    new_Sigma = np.diag(var_w.reshape(-1, )) * 0

    relevance = None
    all_objective_vals = [np.zeros((n_samples, n_iter)) for i in range(n_relfun)]  # list with n_relfun matrices of shape = (n_samples, n_iters)
    # A value is computed for each objective, sample and iteration.

    reward_weight = 0

    epsilon = 0.001  # For detecting convergence
    sliding_window_size = 5  # For detecting convergence
    converged = False

    for iteration in range(n_iter):
        #print(iteration)

        # Verify convergence and if solution is good enough
        if iteration >= sliding_window_size:
            converged = True
            for rel_fun in range(n_relfun):
                current_objective_vals = all_objective_vals[rel_fun]
                objectives_window = current_objective_vals[:, iteration-sliding_window_size:iteration]
                max_val = np.max(objectives_window)
                min_val = np.min(objectives_window)
                if max_val - min_val > epsilon:
                    converged = False

                # Is the solution good enough?
                if rel_fun == 0 or rel_fun == 2:
                    mean_objective_val = np.mean(current_objective_vals[:, iteration-1])
                    if mean_objective_val > 0.005:
                        converged = False
                if rel_fun == 1:
                    min_objective_val = np.min(current_objective_vals[:, iteration - 1])
                    if -min_objective_val < 0.1:
                        converged = False

        if converged:
            print('Convergence achieved')
            # prune all_objective_vals
            for rel_fun in range(n_relfun):
                all_objective_vals[rel_fun] = all_objective_vals[rel_fun][:, 0:iteration+1]
            break

        relevance = learn_relevances(viapoints, start, end, wall_corners, sample_trajectories, n_samples, mu_w, var_w, num_gaba, block_PSI)

        # Iterate over the number of relevance functions
        for rel_fun in range(n_relfun):

            previous_var_w = var_w.reshape(-1, )

            weights_rel = var_w.reshape(-1, ) * relevance[:, rel_fun]  # Compute variances with relevance function.

            weight_samples, traj_samples = sample_trajectories(n_samples, mu_w, weights_rel.reshape(-1, 1), num_gaba, block_PSI)  # Sample weights and trajectories using rescaled variances.

            if rel_fun == 0:  # Relevance function with respect to the start
                objective_val = compute_dist(traj_samples, viapoints[0], start)

            elif rel_fun == 1:  # Relevance function with respect to the minimum signed Euclidean distance to obstacles
                objective_val = -compute_min_s_ED_trajectories(traj_samples, wall_corners)

            elif rel_fun == 2:  # Relevance function with respect to the end
                objective_val = compute_dist(traj_samples, viapoints[1], end)

            elif rel_fun == 3:  # Relevance function with respect to velocity_x
                objective_val = compute_max_velocity(traj_samples, 0)*1e2

            elif rel_fun == 4:  # Relevance function with respect to velocity_y
                objective_val = compute_max_velocity(traj_samples, 1)*1e2

            # Append current objective value
            all_objective_vals[rel_fun][:, iteration] = objective_val.reshape(n_samples, )

            R = -objective_val

            reward_weight = np.exp(beta * R)  # (n_samples, 1)

            # Update distribution
            new_mu_w = np.dot(np.transpose(weight_samples), reward_weight)  # (N*dofs, 1) = (N*dofs, n_samples) * (n_samples, 1)
            W_minus_mu = np.subtract(weight_samples, np.transpose(mu_w))  # (n_samples, N*dofs)
            new_Sigma = new_Sigma * 0  # (N*dofs, N*dofs)

            for i in range(n_samples):
                # new_Sigma = new_Sigma + reward_weight[i] * W_minus_mu[i, :].T * W_minus_mu[i, :]
                new_Sigma = new_Sigma + reward_weight[i] * np.dot(W_minus_mu[i, :].reshape(-1, 1), W_minus_mu[i, :].reshape(1, -1))

            new_mu_w = new_mu_w / np.sum(reward_weight)
            new_Sigma = new_Sigma / np.sum(reward_weight)

            mu_w = new_mu_w
            Sigma_w = new_Sigma
            # Sigma_w = (Sigma_w + Sigma_w.T) / 2  # To make sure Sigma_w is symmetric

            updated_var_w = (1 - relevance[:, rel_fun]) * previous_var_w + \
                relevance[:, rel_fun] * np.diag(Sigma_w)
            var_w = updated_var_w.reshape(-1, 1)


    # Sample from the distribution
    # shape = (n_samples, DOFS*number_features)
    weight_samples, traj_samples = sample_trajectories(5, mu_w, var_w, num_gaba, block_PSI)

    return Learnt_pol(all_objective_vals, traj_samples, relevance, mu_w, var_w), converged


# Learns relevance functions
def learn_relevances(viapoints, start, end, wall_corners, sample_trajectories, n_samples, mu_w, var_w, n_gbf, block_PSI):
    n_relfun = len(viapoints) + 1

    # Sample weights and trajectories
    # trajectory samples: n_samples x max_length*dofs
    # x-vals: 0 --> max_length-1; y-vals: max_length --> 2*max_length-1
    variances = var_w * 0 + 1e-5
    weight_samples, traj_samples = sample_trajectories(n_samples, mu_w, variances, n_gbf, block_PSI)

    # Compute the distances to the start and end positions as well as the minimum signed Euclidean distance to obstacles.
    objective_vals = []
    for rel_fun in range(n_relfun):
        if rel_fun == 0:  # Relevance with respect to distance to the start
            curr_objective_val = compute_dist(traj_samples, viapoints[0], start)
        elif rel_fun == 1:  # Relevance with respect to the minimum signed Euclidean distance to obstacles
            curr_objective_val = -compute_min_s_ED_trajectories(traj_samples, wall_corners)
        elif rel_fun == 2:  # Relevance with respect to distance to end
            curr_objective_val = compute_dist(traj_samples, viapoints[1], end)
        elif rel_fun == 3:  # Relevance function with respect to velocity_x
                objective_val = compute_max_velocity(traj_samples, 0)*1e2
        elif rel_fun == 4:  # Relevance function with respect to velocity_y
                objective_val = compute_max_velocity(traj_samples, 1)*1e2

        objective_vals = curr_objective_val if objective_vals == [] else np.hstack((objective_vals, curr_objective_val))  # (n_samples, number of objectives)

    # Compute Pearson correlation coefficient
    cols_w = weight_samples.shape[1]  # N*dofs
    cols_d = objective_vals.shape[1]  # number of objectives
    relevances = np.zeros((cols_w, cols_d))  # (N*dofs, number of objectives)
    for i in range(cols_w):  # for each weight
        for j in range(cols_d): # for each objective
            relevances[i][j] = np.corrcoef(weight_samples[:, i], objective_vals[:, j], rowvar=False)[0][1]  # QUESTION: Is it possible to vectorize this to get rid of the loop?
    # Normalize s.t. biggest value is 1
    relevances = np.abs(relevances)
    relevances = relevances / relevances.max(axis=0)

    #print("Done calculating relevances")
    return relevances


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


# Computes the x,y length or the theta length of each trajectory
def compute_lens(trajectory_samples_from_promp, n_samples, max_length, dofs, theta=False):
    lens = np.zeros((n_samples, 1))
    for traj_sample in range(n_samples):
        trajectory = trajectory_samples_from_promp[traj_sample, :].reshape((max_length, dofs), order='F')
        if theta == False:
            trajectory = trajectory[:, [0, 1]]
        else:
            trajectory = trajectory[:, 2].reshape(max_length, 1)
        velocities = np.diff(trajectory, axis=0)
        lens[traj_sample, 0] = np.sum(np.sqrt(np.sum(velocities ** 2, axis=1, keepdims=True)))
    return lens  # (n_samples, 1)


# Computes the max velocity for a dof for each sampled trajectory
def compute_max_velocity(traj_samples, dof):  # dof=0 if x; dof=1 if y
    n_samples = traj_samples.shape[0]
    n_dofs = 2
    max_length = int(traj_samples.shape[1] / n_dofs)

    max_velocities = np.zeros((n_samples, 1))

    for i in range(n_samples):
        traj = traj_samples[i, dof*max_length:dof*max_length+max_length]
        velocities = np.abs(np.diff(traj))
        max_velocities[i] = np.max(velocities)

    return max_velocities

