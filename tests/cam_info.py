import math

WIDTH_CONST = 22
HEIGHT_CONST = 22

def average_player_width(width_vec):
    n = len(width_vec)
    width_sum = 0
    for x in width_vec:
        width_sum += x
    return width_sum/n

def average_player_height(height_vec):
    n = len(height_vec)
    height_sum = 0
    for x in height_vec:
        height_sum += x
    return height_sum/n

#return 0 if it is centre cam, else return 1
def cam_placement_type(line):
    k = abs((line[0][0]-line[1][0])/(line[0][1]-line[1][1]))
    return (k < 1)

def get_field_of_view_horizontal(avg_width):
    return WIDTH_CONST/avg_width

def get_field_of_view_vertical(avg_height):
    return HEIGHT_CONST/avg_height

def get_scaled(ang):
    return 2*math.tan(ang/2)
