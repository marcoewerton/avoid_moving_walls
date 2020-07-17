import numpy as np

def PD_controller(point, goal):

    ## point = np.array([x, y, dx, dy])
    ## goal = np.array([xg, yg, dxg, dyg])
    ##
    ## returns acceleration = np.array([ddx, ddy])

    Kpx = 1000
    Kpy = 1000
    Kdx = 30
    Kdy = 30

    x = point[0]
    y = point[1]
    dx = point[2]
    dy = point[3]

    xg = goal[0]
    yg = goal[1]
    dxg = goal[2]
    dyg = goal[3]

    ddx = Kpx*(xg - x) + Kdx*(dxg - dx)
    ddy = Kpy * (yg - y) + Kdy * (dyg - dy)

    return np.array([ddx, ddy])

