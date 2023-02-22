import math
import numpy as np

def rot_around_x(y_pl, y_cen, y_cor):
    a_pl = math.atan(abs(y_pl-y_cen))
    a_cor = math.atan(abs(y_cor-y_cen))
    ang = a_pl + a_cor
    rev = 0
    if abs(y_pl-y_cen) > abs(y_pl-y_cen):
        ang = abs(a_pl-a_cor)
    if y_pl < y_cor:
        rev = 1
    return ang,rev

def rot_around_y(x_pl, x_cen, x_cor):
    a_pl = math.atan(abs(x_pl-x_cen))
    a_cor = math.atan(abs(x_cor-x_cen))
    ang = a_pl + a_cor
    rev = 0
    if abs(x_pl-x_cen) > abs(x_pl-x_cen):
        ang = abs(a_pl-a_cor)
    if x_pl > x_cor:
        rev = 1
    return ang,rev

def get_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
    Px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    Py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    return (Px,Py)

def corner_position(lines):
    return get_intersection(lines[0][0][0],lines[0][0][1],lines[0][1][0],lines[0][1][1],lines[1][0][0],lines[1][0][1],lines[1][1][0],lines[1][1][1])
    

#assumes coordinates scaled to pinhole distance 1,
#write function that scales them!
def get_matrix_rotation(center, corner, player):
    a_x, b_x = rot_around_x(player[1], center[1], corner[1])
    Rx = [[1, 0, 0], [0, math.cos(a_x), -math.sin(a_x)], [0, math.sin(a_x), math.cos(a_x)]]
    if b_x:
        Rx = [[1, 0, 0], [0, math.cos(a_x), math.sin(a_x)], [0, -math.sin(a_x), math.cos(a_x)]]
    a_y, b_y = rot_around_y(player[0], center[0], corner[0])
    Ry = [[math.cos(a_y), 0, math.sin(a_y)], [0, 1, 0], [-math.sin(a_y), 0, math.cos(a_y)]]
    if b_y:
        Ry = [[math.cos(a_y), 0, -math.sin(a_y)], [0, 1, 0], [math.sin(a_y), 0, math.cos(a_y)]]
    R = np.matmul(Rx,Ry)
    return R

def intersection_with_xz(pinhole, vec):
    t = pinhole[1]/vec[1]
    point = [pinhole[0]-t*vec[0], pinhole[1]-t*vec[1], pinhole[2]-t*vec[2]]
    return point
