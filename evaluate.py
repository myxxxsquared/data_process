import numpy as np
import math

EVALUATE_DIR = '/home/rjq/data_cleaned/data_cleaned/evaluate/'
def get_l2_dist(point1, point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5

def evaluate(img, cnts, is_text_cnts, maps, is_va):
    '''
    :param img: ndarrray, np.uint8,
    :param cnts:
        if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2]
        if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
    :param is_text_cnts: bool
    :param maps:
        maps: [TR, TCL, radius, cos_theta, sin_theta], all of them are 2-d array,
        TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
    :return:

    '''
    assert img.shape[:2] == maps[0].shape
    row, col = img.shape[:2]
    [TR, TCL, radius, cos_theta, sin_theta] = maps
    
    # use TR to crop TCL
    cropped_TCL = np.bitwise_and(TR, TCL)

    # pick out instance TCL from cropped_TCL map
    instances = []
    direction_x =[-1,0,1]
    direction_y =[-1,0,1]

    cropped_TCL_for_search = cropped_TCL.copy()
    while np.sum(cropped_TCL_for_search) != 0:
        instance = []
        queue = []
        xs, ys = np.nonzero(cropped_TCL_for_search)
        queue.append((xs[0], ys[0]))
        cropped_TCL_for_search[xs[0],ys[0]] = False
        while len(queue) != 0:
            x, y = queue.pop(0)
            instance.append((x,y))
            for i in range(len(direction_x)):
                for j in range(len(direction_y)):
                    x_next = x+i
                    y_next = y+j
                    if x_next < row and y_next < col and \
                            cropped_TCL_for_search[x_next, y_next] == True:
                        queue.append((x_next, y_next))
                        cropped_TCL_for_search[x_next, y_next] = False
        instances.append(instance)

    # for each instance build its bounding box(represented by cnt)
    cnts = []
    for instance in instances:
        zeros = np.zeros((row, col))
        for x, y in instance:
            r = radius[x, y]
            for i in range(int(r)+1):
                for j in range(int(r)+1):
                    next_x, next_y = x+i, y+j
                    if next_x < row and next_y < col and \
                        get_l2_dist((next_x, next_y), (x, y)) <= r:
                        zeros[next_x, next_y] = 1







