import numpy as np

def compute_min_s_ED(point, wall_corners):
    # point = x,y             ndarray 2x1
    # wall_corners: the 8 corners defining the wall           list of ndarrays. Each ndarray is 2x1

    if point[0] <= wall_corners[0][0]:
        if point[1] >= wall_corners[1][1] or point[1] <= wall_corners[4][1]:
            min_s_ED = wall_corners[0][0] - point[0]
        else:
            tmp1 = np.linalg.norm(wall_corners[1] - point)
            tmp2 = np.linalg.norm(wall_corners[4] - point)
            min_s_ED = min(tmp1, tmp2)
    elif point[0] >= wall_corners[2][0]:
        if point[1] >= wall_corners[2][1] or point[1] <= wall_corners[7][1]:
            min_s_ED = point[0] - wall_corners[2][0]
        else:
            tmp1 = np.linalg.norm(wall_corners[2] - point)
            tmp2 = np.linalg.norm(wall_corners[7] - point)
            min_s_ED = min(tmp1, tmp2)
    else:
        if point[1] >= wall_corners[1][1]:
            min_s_ED = wall_corners[1][1] - point[1]
            # tmp1 = wall_corners[1][0] - point[0]
            # tmp2 = point[0] - wall_corners[2][0]
            # tmp3 = wall_corners[1][1] - point[1]
            # min_s_ED = min(tmp1, tmp2, tmp3)
        elif point[1] <= wall_corners[4][1]:
            min_s_ED = point[1] - wall_corners[4][1]
            # tmp1 = wall_corners[4][0] - point[0]
            # tmp2 = point[0] - wall_corners[7][0]
            # tmp3 = point[1] - wall_corners[4][1]
            # min_s_ED = min(tmp1, tmp2, tmp3)
        else:
            tmp1 = wall_corners[1][1] - point[1]
            tmp2 = point[1] - wall_corners[4][1]
            min_s_ED = min(tmp1, tmp2)

    return float(min_s_ED)