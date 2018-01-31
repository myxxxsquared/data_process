import numpy as np
import cv2
import math
from random import random
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)

MAP_TYPE = np.float32
PIC_TYPE = np.uint8
THICKNESS = 0.15
NEIGHBOR = 3.0
CROPSKEL = 2.0
DIST = 'l2'

def get_l2_dist(point1, point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5

def get_l1_dist(point1, point2):
    return abs(point1[0]-point2[0])+abs(point1[1]-point2[1])

def get_theta(points_list):
    xs, ys = [], []
    for (x,y) in points_list:
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    m,b = np.polyfit(xs, ys, 1)
    theta = np.arctan(m)
    return theta

def is_validate_cnts(im, cnts):
    cols, rows = [], []
    for cnt in cnts:
        for i in range(len(cnt)):
            cols.append(cnt[i][0][0])
            rows.append(cnt[i][0][1])
    col_max = max(cols)
    row_max = max(rows)
    im_row, im_col = im.shape[0]-1, im.shape[1]-1
    flag = True
    if im_row < row_max: flag = False
    if im_col < col_max: flag = False
    return flag

def is_validate_point(im, point):
    row, col = im.shape[:2]
    return (point[0] < row) and (point[1] < col)

def is_inside_point_cnt(point, cnt):
    # ugly place. point here is (row, col)
    # but in the contour points points are (col, row)
    point = (point[1], point[0])
    return cv2.pointPolygonTest(cnt, point, False) >= 0

def validate(im, cnts):
    cols, rows = [], []
    for cnt in cnts:
        for i in range(len(cnt)):
            cols.append(cnt[i][0][0])
            rows.append(cnt[i][0][1])
    col_max = max(cols)
    row_max = max(rows)
    im_row, im_col = im.shape[0]-1, im.shape[1]-1
    if im_row < row_max:
        temp = np.zeros([row_max-im_row, im.shape[1], im.shape[2]])
        im = np.concatenate((im, temp), 0)
    if im_col < col_max:
        temp = np.zeros([im.shape[0], col_max-im_col, im.shape[2]])
        im = np.concatenate((im, temp), 1)
    return im, cnts

def resize(im, cnts, row, col):
    im_row, im_col = im.shape[0], im.shape[1]
    im_ = cv2.resize(im, (row, col), interpolation = cv2.INTER_CUBIC)
    cnts_ = []
    for cnt in cnts:
        temp = np.zeros_like(cnt)
        for i in range(len(cnt)):
            temp[i][0][0] = int(cnt[i][0][0]*col/im_col)
            temp[i][0][1] = int(cnt[i][0][1]*row/im_row)
        cnts_.append(temp)
    return im_, cnts_

def find_mid_line_and_radius(points_list,dist='l1',sampling_num=500):

    def neg_cosine(p1,p2,p3,p4):
        vector_1 = (p2[0]-p1[0],p2[1]-p1[1])
        vector_2 = (p4[0]-p3[0], p4[1]-p3[1])
        if get_l2_dist(vector_2,(0,0))*get_l2_dist(vector_1,(0,0)) == 0.0:
            return 0.0
        return (vector_1[1]*vector_2[1]+vector_1[0]*vector_2[0])/(get_l2_dist(vector_2,(0,0))*get_l2_dist(vector_1,(0,0)))

    def sampling(p1,p2,sampling_num):
        x = np.linspace(p1[0], p2[0], sampling_num)
        y = np.linspace(p1[1], p2[1], sampling_num)
        return [(x[i],y[i]) for i in range(x.shape[0])]

    if dist == 'l1':
        dist_func = get_l1_dist
    elif dist == 'l2':
        dist_func = get_l2_dist
    else:
        raise NotImplementedError(dist+' is not implemented')

    points_list = [tuple(point) for point in points_list]

    if len(points_list) > 4:
        circular_points_list=points_list+points_list[:3]
        cosine_list=[(circular_points_list[i+1],circular_points_list[i+2],
                      neg_cosine(circular_points_list[i+0],
                                 circular_points_list[i+1],
                                 circular_points_list[i+2],
                                 circular_points_list[i+3]),i)
                     for i in range(len(points_list))]
        best=None
        #(i,j),i<j
        for i in range(len(cosine_list)):
            for j in range(i+1,len(cosine_list)):
                if cosine_list[i][0]!=cosine_list[j][1] and cosine_list[j][0]!=cosine_list[i][1]:
                    if best is None:
                        best=(i,j)
                    else:
                        if cosine_list[best[0]][2]+cosine_list[best[1]][2]>cosine_list[i][2]+cosine_list[j][2]:
                            best = (min(i,j), max(i,j))

        line_one = [cosine_list[p][1] for p in range(best[0],best[1])]
        line_two = ([cosine_list[p][1] for p in range(best[1],len(cosine_list))]+[cosine_list[p][1] for p in range(0, best[0])])[::-1]
    elif len(points_list) == 4:
        d1=get_l2_dist(points_list[0],points_list[1])
        d2 = get_l2_dist(points_list[1], points_list[2])
        d3 = get_l2_dist(points_list[2], points_list[3])
        d4 = get_l2_dist(points_list[3], points_list[0])
        if d1+d3<d2+d4:
            line_one = [points_list[1], points_list[2]]
            line_two = [points_list[0], points_list[3]]
        else:
            line_one = [points_list[0], points_list[1]]
            line_two = [points_list[3], points_list[2]]
    elif len(points_list) == 3:
        d1 = get_l2_dist(points_list[0], points_list[1])
        d2 = get_l2_dist(points_list[1], points_list[2])
        d3 = get_l2_dist(points_list[2], points_list[0])

        if d1 < min(d2, d3):
            line_one = [points_list[2], points_list[0]]
            line_two = [points_list[2], points_list[1]]
        if d2 < min(d3, d1):
            line_one = [points_list[0], points_list[1]]
            line_two = [points_list[0], points_list[2]]
        if d3 < min(d2, d1):
            line_one = [points_list[1], points_list[0]]
            line_two = [points_list[1], points_list[2]]
    else:
        raise AttributeError('points_list with less than 3 points')

    point_list_one = []
    point_list_two = []
    total_len_one=sum([get_l2_dist(line_one[i],line_one[i+1]) for i in range(len(line_one)-1)])
    total_len_two=sum([get_l2_dist(line_two[i], line_two[i + 1]) for i in range(len(line_two) - 1)])
    for p in range(len(line_one)-1):
        point_list_one+=sampling(line_one[p+0],line_one[p+1],math.floor(get_l2_dist(line_one[p+0],line_one[p+1])*sampling_num/total_len_one))

    for p in range(len(line_two) - 1):
        point_list_two+=sampling(line_two[p+0],line_two[p+1],math.floor(get_l2_dist(line_two[p+0],line_two[p+1])*sampling_num/total_len_two))

    if len(point_list_one)!=len(point_list_two):
        if len(point_list_one)<len(point_list_two):
            point_list_one,point_list_two=point_list_two,point_list_one #by default, one > two
        while len(point_list_one)!=len(point_list_two):
            p=int(random()*len(point_list_one))
            point_list_one.pop(p)

    center_line = []
    radius_dict = {}
    theta_dict = {}

    for i in range(len(point_list_one)):
        x1, y1 = point_list_one[i][0], point_list_one[i][1]
        x2, y2 = point_list_two[i][0], point_list_two[i][1]
        x, y = int(round((x1+x2)/2)), int(round((y1+y2)/2))
        center_line.append((x,y))
        radius_dict[(x,y)] = dist_func((x1,y1),(x2,y2))/2

    if len(points_list) == 4:
        theta = get_theta(center_line)
        for point in center_line:
            theta_dict[point] = theta
    else:
        for point in center_line:
            width = int(NEIGHBOR * radius_dict[point])
            fit_points = []
            for fit_point in center_line:
                if point[0] - width <= fit_point[0] <= point[0] + width and \
                        point[1] - width <= fit_point[1] <= point[1] + width:
                    fit_points.append(fit_point)
            theta = get_theta(fit_points)
            theta_dict[point] = theta

    temp = []
    temp_radius_dict = {}
    temp_theta_dict = {}
    crop_length1 = radius_dict[center_line[0]]
    crop_length2 = radius_dict[center_line[-1]]
    decrease = 0.0
    while len(temp) <= 1:
        for point in center_line:
            crop_length1 = crop_length1*(1+decrease)
            crop_length2 = crop_length2*(1+decrease)
            if dist_func(point, center_line[0]) >= crop_length1*CROPSKEL and \
               dist_func(point, center_line[-1]) >= crop_length2 * CROPSKEL:
                temp.append(point)
                temp_radius_dict[point] = radius_dict[point]
                temp_theta_dict[point] = theta_dict[point]
        decrease += 0.01
    center_line = temp
    radius_dict = temp_radius_dict
    theta_dict = temp_theta_dict
    return center_line, radius_dict, theta_dict


def get_maps_algo3(im, cnts):
    global error
    if DIST == 'l1': dist_func = get_l1_dist
    elif DIST == 'l2': dist_func = get_l2_dist
    else: raise NotImplementedError(DIST+' is not implemented')

    skels_points = []
    radius_dict = {}
    score_dict = {}
    curvature_dict = {}
    theta_dict = {}
    mask_fills = []
    for cnt in cnts:
        cnt = np.squeeze(cnt)
        point_list = [(point[1],point[0]) for point in cnt]
        skel_points, radius_dict_cnt, theta_dict_cnt = find_mid_line_and_radius(point_list, dist=DIST, sampling_num=500)

        for point, radius in radius_dict_cnt.items():
            radius_dict[point] = radius
        for point, theta in theta_dict_cnt.items():
            theta_dict[point] = theta

        [skels_points.append(point) for point in skel_points]
        mask_zero = np.zeros(im.shape[:2], dtype = np.uint8)
        mask_fill = cv2.fillPoly(mask_zero, pts = [cnt], color=(255))
        mask_fills.append(np.sign(mask_fill.copy()).astype(MAP_TYPE))

        # get belt
        belt = set()
        connect_dict = {}
        for point in skel_points:
            thickness = int(THICKNESS*radius_dict[point])
            for i in range(-thickness, thickness+1):
                for j in range(-thickness, thickness+1):
                    candidate = (point[0]+i, point[1]+j)
                    if is_validate_point(im, candidate):
                        belt.add(candidate)
                        if candidate not in connect_dict:
                            connect_dict[candidate] = []
                        connect_dict[candidate].append(point)

        # score map
        for point in belt:
            score_dict[point] = 1.0

        # theta, raidus map
        for point in belt:
            min_dist = 1e8
            min_dist_point = None
            for skel_point in connect_dict[point]:
                dist = dist_func(point, skel_point)
                if dist < min_dist:
                    min_dist_point = skel_point
                    min_dist = dist
            theta_dict[point] = theta_dict[min_dist_point[0], min_dist_point[1]]
            radius_dict[point] = radius_dict[min_dist_point[0], min_dist_point[1]]-min_dist

        # curvature map
        for point in belt:
            curvature_dict[tuple(point)] = 0.0

    return skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills

def get_maps(im, cnts, algo = 3):
    if algo == 3:
        skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills = \
            get_maps_algo3(im,cnts)
    else:
        raise NotImplementedError('algo'+str(algo)+'is not implemented')

    mask_skel = np.zeros(im.shape[:2], dtype = PIC_TYPE)
    for point in skels_points:
        mask_skel[point[0],point[1]] = 255

    maps = np.zeros(list(im.shape[:2])+[4], dtype = MAP_TYPE)
    for point, s in score_dict.items():
        maps[point[0],point[1],0] = s
    for point, t in theta_dict.items():
        maps[point[0],point[1],1] = t
    for point, c in curvature_dict.items():
        maps[point[0],point[1],2] = c
    for point, r in radius_dict.items():
        maps[point[0],point[1],3] = r

    mask_fill = np.expand_dims(np.sign(np.sum(mask_fills, 0)).astype(MAP_TYPE),2)
    maps = np.concatenate((maps, mask_fill), 2)
    return mask_skel, maps

