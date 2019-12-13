import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from shapely.geometry import LineString, Point

doom_red_color = [30, 30, 161]
doom_blue_color = [161, 30, 23]
doom_green_color = [48, 161, 40]

DEFAULT_TRAIN_PATH = Path("flat")
SCENARIO_FILE = "height2.wad"
prefix = SCENARIO_FILE.split('.')[0]
if not os.path.exists(DEFAULT_TRAIN_PATH / prefix):
    os.makedirs(DEFAULT_TRAIN_PATH / prefix)

file_no = 0

# lines from left to right
lines_ltr = []

# lines from top to bottom
lines_ttb = []

# 1-D Array of intersection points
intersections = []

# 8-D Array of nodes with label
label_arr = []  # np.ones(80, dtype='bool')

dim_arr = [3, 5, 7, 9, 11, 13, 15, 17]
dim = 3
for i in range(8):
    label_arr.append(np.ones(dim, dtype='bool'))
    dim += 2

labels_to_plot = []

def set_labels(labels):
    global labels_to_plot
    labels_to_plot = labels

# !NUR VERWENDEN WENN MAN DIE DATEN VON JSON EINGELESEN UND IM labels_to_plot ABGESPEICHERT HAT!
def plot_data():
    global intersections
    global labels_to_plot
    print(len(labels_to_plot), ' --- ', len(intersections))
    i = 0

    for p in intersections[::-1]:
        color = 'green' if labels_to_plot[i] else 'red'
        plt.plot(p.x, p.y, marker='o', color=color, mew=0.1)
        i += 1


def flatten(l):
    return [bool(item) for sublist in l for item in sublist]


def save_data(screen):
    global label_arr
    global file_no
    global SCENARIO_FILE
    global DEFAULT_TRAIN_PATH
    global prefix

    if file_no == 0:
        while (DEFAULT_TRAIN_PATH / prefix / f"{prefix}_{file_no}.png").exists():
            file_no += 1

    filename = DEFAULT_TRAIN_PATH / prefix / f"{prefix}_{file_no}"

    with open(str(filename) + '.json', 'w') as json_file:
        json.dump(flatten(label_arr), json_file)

    cv2.imwrite(str(filename) + '.png', screen)
    file_no += 1

    print("Data saved! IMG: ", file_no - 1)


def label(pos_x, pos_y, pos_z, sectors, labels):
    global intersections
    global label_arr
    global dim_arr
    # Init
    label_arr = [[True for _ in range(dim_arr[j])] for j in range(len(dim_arr))]

    # sort from closest to far
    intersections.reverse()
    layer_index = 0
    j = 0
    for node in intersections:
        # line to the node
        view_line = LineString([(pos_x, pos_y), (node.x, node.y)])

        if j == dim_arr[layer_index]:
            layer_index += 1
            j = 0

        if label_arr[layer_index][j]:
            # check the lines if they intersect with view_line
            for s in sectors:
                if s.floor_height == s.ceiling_height:
                    continue
                for l in s.lines:
                    line = LineString([(l.x1, l.y1), (l.x2, l.y2)])
                    inter = line.intersection(view_line)
                    # aus irgendeinem Grund wird eine positive Höhe negativ abgespeichert,
                    # deshalb muss die Höhe zuvor "geflippt" werden
                    if inter and (l.is_blocking or (-1 * s.floor_height - pos_z > 24)):
                        label_arr[layer_index][j] = False
                        m = None
                        if j == dim_arr[layer_index] - 1:
                            m = -1
                        elif j == 0:
                            m = j
                        elif j == dim_arr[layer_index] // 2:
                            m = dim_arr[layer_index] // 2
                        if m is not None:
                            for k in range(layer_index + 1, len(dim_arr)):
                                label_arr[k][m] = False

            # Notwending, wenn blockierende Objekte in der Szene vorhanden sind.

            # # check the objects if they are on the view_line
            # for o in labels:
            #     if o.object_name != "DoomPlayer" and o.object_name != "Blood" and o.object_position_z == pos_z:
            #         point = Point(o.object_position_x, o.object_position_y)
            #         if view_line.distance(point) < 5:
            #             label_arr[layer_index][j] = False
        j += 1


def plot_intersections():
    global intersections
    global dim_arr
    layer_index = 0
    i = 0
    intersections.reverse()
    for p in intersections:
        if i == dim_arr[layer_index]:
            layer_index += 1
            i = 0
        color = 'green' if label_arr[layer_index][i] else 'red'
        plt.plot(p.x, p.y, marker='o', color=color, mew=0.1, label='nodes')
        i += 1


def line_intersection(l1, l2):
    line1 = LineString(l1)
    line2 = LineString(l2)

    return line1.intersection(line2)


def create_intersection_points():
    global intersections
    global lines_ltr
    global lines_ttb

    for l1 in lines_ttb:
        for l2 in lines_ltr:
            inter = line_intersection(l1, l2)
            if inter:
                intersections.append(inter)


# Print information about sectors.
def print_sectors(sectors):
    # print("Sectors:")
    for s in sectors:
        print("Sector floor height:", s.floor_height, "ceiling height:", s.ceiling_height)
        print("Sector lines:", [(l.x1, l.y1, l.x2, l.y2, l.is_blocking) for l in s.lines if (s.floor_height == -52)])

        # Plot sector on map
        for l in s.lines:
            if l.is_blocking:
                plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)
            else:
                plt.plot([l.x1, l.x2], [l.y1, l.y2], color='blue', linewidth=1)


def print_objects(objects, pos_x, pos_y, pos_z):
    for i, o in enumerate(objects):
        if o.name == "DoomPlayer":
            # print("Object position x:", o.position_x, "y:", o.position_y, "z:", o.position_z)

            # Other available fields:
            # print("Object rotation angle", o.angle, "pitch:", o.pitch, "roll:", o.roll)
            # print("Object velocity x:", o.velocity_x, "y:", o.velocity_y, "z:", o.velocity_z)
            # marker=(3, 0, o.angle + 45)
            plt.plot(o.position_x, o.position_y, color='#000099', marker='o')

        else:
            # print("Object name:", o.name)
            # print("Distance to Object: ", o.name)
            # print("Object position x:", o.position_x, "y:", o.position_y, "z:", o.position_z)
            # print("Eucl. Distance: ", calc_distance(pos_x, pos_y, pos_z, o.position_x, o.position_y, o.position_z))
            plt.plot(o.position_x, o.position_y, color='red', marker='.')


def get_fov(pos_x, pos_y, angle, fov, radius):
    half_deg = fov // 2

    x1 = pos_x + math.cos(math.radians(angle + half_deg)) * radius
    y1 = pos_y + math.sin(math.radians(angle + half_deg)) * radius

    x2 = pos_x + math.cos(math.radians(angle - half_deg)) * radius
    y2 = pos_y + math.sin(math.radians(angle - half_deg)) * radius

    return (x1, y1), (x2, y2)


def plot_fov(pos_x, pos_y, p1, p2):
    plt.plot([pos_x, p1[0]], [pos_y, p1[1]], color='#C1C1C1')
    plt.plot([pos_x, p2[0]], [pos_y, p2[1]], color='#C1C1C1')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#AAAAAA')

    create_nodes(pos_x, pos_y, p1[0], p1[1], p2[0], p2[1])
    make_grid(pos_x, pos_y, p1[0], p1[1], p2[0], p2[1])


#  creates the lines to make a grid
def create_nodes(pos_x, pos_y, x1, y1, x2, y2):
    c_mid_x, c_mid_y = mid_point(x1, y1, x2, y2)
    a_mid_x, a_mid_y = mid_point(x1, y1, pos_x, pos_y)
    b_mid_x, b_mid_y = mid_point(x2, y2, pos_x, pos_y)

    c_mid_1 = mid_point(x1, y1, c_mid_x, c_mid_y)
    c_mid_2 = mid_point(x2, y2, c_mid_x, c_mid_y)

    c_mid_11 = mid_point(x1, y1, c_mid_1[0], c_mid_1[1])
    c_mid_12 = mid_point(c_mid_1[0], c_mid_1[1], c_mid_x, c_mid_y)

    c_mid_21 = mid_point(c_mid_2[0], c_mid_2[1], c_mid_x, c_mid_y)
    c_mid_22 = mid_point(x2, y2, c_mid_2[0], c_mid_2[1])
    c_mid_111 = mid_point(c_mid_11[0], c_mid_11[1], x1, y1)
    c_mid_112 = mid_point(c_mid_11[0], c_mid_11[1], c_mid_1[0], c_mid_1[1])
    c_mid_121 = mid_point(c_mid_1[0], c_mid_1[1], c_mid_12[0], c_mid_12[1])
    c_mid_122 = mid_point(c_mid_12[0], c_mid_12[1], c_mid_x, c_mid_y)
    c_mid_211 = mid_point(c_mid_21[0], c_mid_21[1], c_mid_x, c_mid_y)
    c_mid_212 = mid_point(c_mid_21[0], c_mid_21[1], c_mid_2[0], c_mid_2[1])
    c_mid_221 = mid_point(c_mid_22[0], c_mid_22[1], c_mid_2[0], c_mid_2[1])
    c_mid_222 = mid_point(c_mid_22[0], c_mid_22[1], x2, y2)

    a_mid_1 = mid_point(x1, y1, a_mid_x, a_mid_y)
    a_mid_2 = mid_point(pos_x, pos_y, a_mid_x, a_mid_y)
    a_mid_11 = mid_point(x1, y1, a_mid_1[0], a_mid_1[1])
    a_mid_12 = mid_point(a_mid_1[0], a_mid_1[1], a_mid_x, a_mid_y)
    a_mid_21 = mid_point(a_mid_2[0], a_mid_2[1], a_mid_x, a_mid_y)
    a_mid_22 = mid_point(pos_x, pos_y, a_mid_2[0], a_mid_2[1])

    b_mid_2 = mid_point(pos_x, pos_y, b_mid_x, b_mid_y)
    b_mid_1 = mid_point(x2, y2, b_mid_x, b_mid_y)
    b_mid_11 = mid_point(x2, y2, b_mid_1[0], b_mid_1[1])
    b_mid_12 = mid_point(b_mid_1[0], b_mid_1[1], b_mid_x, b_mid_y)
    b_mid_21 = mid_point(b_mid_2[0], b_mid_2[1], b_mid_x, b_mid_y)
    b_mid_22 = mid_point(pos_x, pos_y, b_mid_2[0], b_mid_2[1])

    # LINES to get the intersection

    # lines left to right (in order)
    line_ltr_11 = [a_mid_11, c_mid_111]
    line_ltr_1 = [a_mid_1, c_mid_11]
    line_ltr_12 = [a_mid_12, c_mid_112]
    line_ltr_2 = [(a_mid_x, a_mid_y), c_mid_1]
    line_ltr_31 = [a_mid_21, c_mid_121]
    line_ltr_3 = [a_mid_2, c_mid_12]
    line_ltr_32 = [a_mid_22, c_mid_122]
    line_ltr_4 = [(pos_x, pos_y), (c_mid_x, c_mid_y)]
    line_ltr_51 = [b_mid_22, c_mid_211]
    line_ltr_5 = [b_mid_2, c_mid_21]
    line_ltr_52 = [b_mid_21, c_mid_212]
    line_ltr_6 = [(b_mid_x, b_mid_y), c_mid_2]
    line_ltr_71 = [b_mid_12, c_mid_221]
    line_ltr_7 = [b_mid_1, c_mid_22]
    line_ltr_72 = [b_mid_11, c_mid_222]

    global lines_ltr
    lines_ltr = [line_ltr_11, line_ltr_1, line_ltr_12, line_ltr_2, line_ltr_31, line_ltr_3, line_ltr_32, line_ltr_4,
                 line_ltr_51, line_ltr_5, line_ltr_52, line_ltr_6, line_ltr_71, line_ltr_7, line_ltr_72
                 ]

    # lines top to bottom (from far to close)
    line_ttb_11 = [a_mid_11, b_mid_11]
    line_ttb_1 = [a_mid_1, b_mid_1]
    line_ttb_12 = [a_mid_12, b_mid_12]
    line_ttb_2 = [(a_mid_x, a_mid_y), (b_mid_x, b_mid_y)]
    line_ttb_31 = [a_mid_21, b_mid_21]
    line_ttb_3 = [a_mid_2, b_mid_2]
    line_ttb_32 = [a_mid_22, b_mid_22]

    global lines_ttb
    lines_ttb = [line_ttb_11, line_ttb_1, line_ttb_12, line_ttb_2, line_ttb_31, line_ttb_3, line_ttb_32]

    global intersections

    intersections = [Point(x1, y1), Point(c_mid_111), Point(c_mid_11), Point(c_mid_112), Point(c_mid_1),
                     Point(c_mid_121), Point(c_mid_12), Point(c_mid_122), Point(c_mid_x, c_mid_y),
                     Point(c_mid_211), Point(c_mid_21), Point(c_mid_212), Point(c_mid_2), Point(c_mid_221),
                     Point(c_mid_22), Point(c_mid_222),
                     Point(x2, y2)]

    create_intersection_points()


'''  x1
    /|
 a / |
  /  |
p    | c
  \  |
 b \ |
    \|
     x2
    make_grids to a triangle
    needs 3 points
'''


def make_grid(pos_x, pos_y, x1, y1, x2, y2):
    c_mid_x, c_mid_y = mid_point(x1, y1, x2, y2)
    a_mid_x, a_mid_y = mid_point(x1, y1, pos_x, pos_y)
    b_mid_x, b_mid_y = mid_point(x2, y2, pos_x, pos_y)

    c_mid_1 = mid_point(x1, y1, c_mid_x, c_mid_y)
    c_mid_2 = mid_point(x2, y2, c_mid_x, c_mid_y)

    c_mid_11 = mid_point(x1, y1, c_mid_1[0], c_mid_1[1])
    c_mid_111 = mid_point(c_mid_11[0], c_mid_11[1], x1, y1)
    c_mid_112 = mid_point(c_mid_11[0], c_mid_11[1], c_mid_1[0], c_mid_1[1])

    c_mid_12 = mid_point(c_mid_1[0], c_mid_1[1], c_mid_x, c_mid_y)
    c_mid_121 = mid_point(c_mid_1[0], c_mid_1[1], c_mid_12[0], c_mid_12[1])
    c_mid_122 = mid_point(c_mid_12[0], c_mid_12[1], c_mid_x, c_mid_y)

    c_mid_21 = mid_point(c_mid_2[0], c_mid_2[1], c_mid_x, c_mid_y)
    c_mid_211 = mid_point(c_mid_21[0], c_mid_21[1], c_mid_x, c_mid_y)
    c_mid_212 = mid_point(c_mid_21[0], c_mid_21[1], c_mid_2[0], c_mid_2[1])

    c_mid_22 = mid_point(x2, y2, c_mid_2[0], c_mid_2[1])
    c_mid_221 = mid_point(c_mid_22[0], c_mid_22[1], c_mid_2[0], c_mid_2[1])
    c_mid_222 = mid_point(c_mid_22[0], c_mid_22[1], x2, y2)

    a_mid_1 = mid_point(x1, y1, a_mid_x, a_mid_y)
    a_mid_2 = mid_point(pos_x, pos_y, a_mid_x, a_mid_y)
    a_mid_11 = mid_point(x1, y1, a_mid_1[0], a_mid_1[1])
    a_mid_12 = mid_point(a_mid_1[0], a_mid_1[1], a_mid_x, a_mid_y)
    a_mid_21 = mid_point(a_mid_2[0], a_mid_2[1], a_mid_x, a_mid_y)
    a_mid_22 = mid_point(pos_x, pos_y, a_mid_2[0], a_mid_2[1])

    b_mid_2 = mid_point(pos_x, pos_y, b_mid_x, b_mid_y)
    b_mid_1 = mid_point(x2, y2, b_mid_x, b_mid_y)
    b_mid_11 = mid_point(x2, y2, b_mid_1[0], b_mid_1[1])
    b_mid_12 = mid_point(b_mid_1[0], b_mid_1[1], b_mid_x, b_mid_y)
    b_mid_21 = mid_point(b_mid_2[0], b_mid_2[1], b_mid_x, b_mid_y)
    b_mid_22 = mid_point(pos_x, pos_y, b_mid_2[0], b_mid_2[1])

    # middle line
    plt.plot([c_mid_x, pos_x], [c_mid_y, pos_y], color='#CCCCCC')

    # Lines from right to left
    plt.plot([c_mid_1[0], a_mid_x], [c_mid_1[1], a_mid_y], color='#CCCCCC')
    plt.plot([c_mid_2[0], b_mid_x], [c_mid_2[1], b_mid_y], color='#CCCCCC')

    plt.plot([a_mid_1[0], c_mid_11[0]], [a_mid_1[1], c_mid_11[1]], color='#CCCCCC')
    plt.plot([a_mid_2[0], c_mid_12[0]], [a_mid_2[1], c_mid_12[1]], color='#CCCCCC')

    plt.plot([b_mid_2[0], c_mid_21[0]], [b_mid_2[1], c_mid_21[1]], color='#CCCCCC')
    plt.plot([b_mid_1[0], c_mid_22[0]], [b_mid_1[1], c_mid_22[1]], color='#CCCCCC')

    plt.plot([a_mid_12[0], c_mid_112[0]], [a_mid_12[1], c_mid_112[1]], color='#CCCCCC')
    plt.plot([a_mid_22[0], c_mid_122[0]], [a_mid_22[1], c_mid_122[1]], color='#CCCCCC')
    plt.plot([a_mid_11[0], c_mid_111[0]], [a_mid_11[1], c_mid_111[1]], color='#CCCCCC')
    plt.plot([a_mid_21[0], c_mid_121[0]], [a_mid_21[1], c_mid_121[1]], color='#CCCCCC')

    plt.plot([b_mid_12[0], c_mid_221[0]], [b_mid_12[1], c_mid_221[1]], color='#CCCCCC')
    plt.plot([b_mid_22[0], c_mid_211[0]], [b_mid_22[1], c_mid_211[1]], color='#CCCCCC')
    plt.plot([b_mid_11[0], c_mid_222[0]], [b_mid_11[1], c_mid_222[1]], color='#CCCCCC')
    plt.plot([b_mid_21[0], c_mid_212[0]], [b_mid_21[1], c_mid_212[1]], color='#CCCCCC')

    # Lines from up to down
    plt.plot([a_mid_x, b_mid_x], [a_mid_y, b_mid_y], color='#CCCCCC')
    plt.plot([a_mid_1[0], b_mid_1[0]], [a_mid_1[1], b_mid_1[1]], color='#CCCCCC')
    plt.plot([a_mid_2[0], b_mid_2[0]], [a_mid_2[1], b_mid_2[1]], color='#CCCCCC')
    plt.plot([a_mid_12[0], b_mid_12[0]], [a_mid_12[1], b_mid_12[1]], color='#CCCCCC')
    plt.plot([a_mid_22[0], b_mid_22[0]], [a_mid_22[1], b_mid_22[1]], color='#CCCCCC')
    plt.plot([a_mid_11[0], b_mid_11[0]], [a_mid_11[1], b_mid_11[1]], color='#CCCCCC')
    plt.plot([a_mid_21[0], b_mid_21[0]], [a_mid_21[1], b_mid_21[1]], color='#CCCCCC')


def mid_point(x1, y1, x2, y2):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    return mid_x, mid_y


def plot_objects_in_view(labels):
    for l in labels:
        print("Label:", l.value, "object id:", l.id, "object name:", l.name)
        print("Object position x:", l.position_x, "y:", l.position_y, "z:", l.position_z)
        plt.plot(l.position_x, l.position_y, color='blue', marker='<')

#################################################################
##      Nicht für die Datenerhebung genutzte Funktionen        ##
#################################################################

def save_labels_as_json(labels):
    json_obj = []

    for l in labels:
        tmp = {
            "value": l.value,
            "id": l.id,
            "name": l.name,
            "pos_x": l.position_x,
            "pos_y": l.position_y,
            "pos_z": l.position_z,
        }

        json_obj.append(tmp)

        with open('labels.json', 'w') as json_file:
            json.dump(json_obj, json_file)
            print('should save labels info')


def calc_distance(x1, y1, z1, x2, y2, z2):
    a = np.array([x1, y1, z1])
    b = np.array([x2, y2, z2])

    dist = np.linalg.norm(a - b)

    return dist


# segmented walls, ceilings, and floors
def segmentation(label, pitch):
    tresh = label.shape[0] // 2

    offset = math.floor(pitch * -tresh / 32)

    tmp = np.stack([label] * 3, -1)

    tmp[label == 0] = doom_red_color

    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            # Ceiling ~ red
            if label[y, x] == 1 and y < tresh + offset:
                tmp[y, x] = doom_blue_color
            # Floor ~ green
            elif label[y, x] == 1 and y >= tresh + offset:
                tmp[y, x] = doom_green_color
    return tmp


def color_labels(labels):
    """
    Walls are blue, floor/ceiling are red (OpenCV uses BGR).
    """
    tmp = np.stack([labels] * 3, -1)
    tmp[labels == 0] = [255, 0, 0]
    tmp[labels == 1] = [0, 0, 255]

    return tmp
