import numpy as np

def plant(acceleration, point, dt):

    ddx = acceleration[0]
    ddy = acceleration[1]
    x0 = point[0]
    y0 = point[1]
    dx0 = point[2]
    dy0 = point[3]

    dx = dx0 + ddx*dt
    dy = dy0 + ddy*dt

    x = x0 + dx0*dt + 0.5*ddx*dt**2
    y = y0 + dy0*dt + 0.5*ddy*dt**2

    return np.array([x, y, dx, dy])

