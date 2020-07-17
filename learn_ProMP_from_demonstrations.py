import numpy as np
import ProMP as pmp


def learn_promp(num_gaba, n_demonstrations, folder="Demonstrations"):

    # load demonstrated trajectories
    demonstrations = []  # A list of ndarrays
    for index in range(n_demonstrations):
        string = folder + "/traj" + str(index) + ".out"
        traj = np.genfromtxt(string, delimiter=',')
        demonstrations.append(traj)

    # learn ProMP
    myProMP = pmp.ProMP(demonstrations, num_gaba)  # The trajectories in demonstrations are time-aligned by ProMP.

    return myProMP
