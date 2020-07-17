import matplotlib.pyplot as plt
import numpy as np

trajectories = []

# load trajectories from text files
for i in range(10):
    string = "Demonstrations/" + "traj" + str(i) + ".out"
    traj = np.genfromtxt(string, delimiter=',')
    trajectories.append(traj)

# plot trajectories
for i in range(10):
    plt.plot(trajectories[i][:, 0], trajectories[i][:, 1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Demonstrations')
plt.show()


