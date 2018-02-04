import numpy as np
from skimage import morphology
import cv2
import math
from random import random
'''
1. im is a image, type is numpy.ndarray with shape of (row, col, channel)
2. cnts is a list. Its elements are numpy.ndarray with shape of (n, 1, 2) and with dtype of np.int32
3. point is a tuple (x, y), x denotes the row of a image, y denotes the col of a image, therefore im[x, y] are the pixels of point (x,y)
'''

MAP_TYPE = np.float32
PIC_TYPE = np.uint8
THICKNESS = 0.2
NEIGHBOR = 1.0
FIND_MID = 0.3


def get_l2_dist(point1, point2):
    return np.sqrt(np.square(point1[0]-point2[0])+np.square(point1[1]-point2[1])).astype(MAP_TYPE)

def get_theta(points_list):
    xs, ys = [], []
    for (x,y) in points_list:
        xs.append((x, 1))
        ys.append(y)
    xs = np.array(xs).astype(np.float64)
    ys = np.array(ys).astype(np.float64)
    #k = (len(xs)*np.sum(xs*ys) - np.sum(xs)*np.sum(ys))/(len(xs)*np.sum(np.square(xs))-np.square(np.sum(xs)))
    k = np.linalg.lstsq(xs, ys)[0][0]
    theta = np.arctan(k).astype(MAP_TYPE)
    return theta

def find_mid_line(points_list):
    points_conjunct_dict = {}
    for point in points_list:
        points_conjunct_dict[point] = []
    for point in points_list:
        if (point[0]+1, point[1]) in points_conjunct_dict: points_conjunct_dict[point].append((point[0]+1, point[1]))
        if (point[0]-1, point[1]) in points_conjunct_dict: points_conjunct_dict[point].append((point[0]-1, point[1]))
        if (point[0]+1, point[1]+1) in points_conjunct_dict: points_conjunct_dict[point].append((point[0]+1, point[1]+1))
        if (point[0]+1, point[1]-1) in points_conjunct_dict: points_conjunct_dict[point].append((point[0]+1, point[1]-1))
        if (point[0]-1, point[1]+1) in points_conjunct_dict: points_conjunct_dict[point].append((point[0]-1, point[1]+1))
        if (point[0]-1, point[1]-1) in points_conjunct_dict: points_conjunct_dict[point].append((point[0]-1, point[1]-1))
        if (point[0], point[1]+1) in points_conjunct_dict: points_conjunct_dict[point].append((point[0], point[1]+1))
        if (point[0], point[1]-1) in points_conjunct_dict: points_conjunct_dict[point].append((point[0], point[1]-1))

    mid_line_head_points = []
    for point in points_list:
        if len(points_conjunct_dict[point]) == 3: mid_line_head_points.append(point)

    def find_end(start, toward, points_conjunct_dict):
        path = [start]
        def fn(present, prev):
            if present in points_conjunct_dict and len(points_conjunct_dict[present]) == 2:
                path.append(present)
                assert prev in points_conjunct_dict[present]
                for next_ in points_conjunct_dict[present]:
                    if next_ != prev: return fn(next_, present)
            if present in points_conjunct_dict and len(points_conjunct_dict[present]) != 2:
                path.append(present)
                return present
            if present not in points_conjunct_dict:
                raise AttributeError('find_end wrong')
        end_point = fn(toward, start)
        return path, end_point

    mid_line_points = set()
    for point in mid_line_head_points:
        for toward in points_conjunct_dict[point]:
             path, end_point = find_end(point, toward, points_conjunct_dict)
             if len(points_conjunct_dict[end_point]) == 3:
                 [mid_line_points.add(point) for point in path]
    return list(mid_line_points)

def find_mid_line_with_dist(skel_points, dist_mask):
    pixel_num = np.sum(np.sign(dist_mask))
    belt = set()
    for index in np.argsort(dist_mask, axis=None)[-int(pixel_num*FIND_MID):]:
        belt.add((index//dist_mask.shape[0], index%dist_mask.shape[0]))
    mid_line_points = set()
    for point in skel_points:
        if point in belt:
            mid_line_points.add(point)
    return list(mid_line_points)

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
    return (point[0] < im.shape[0]) and (point[1] < im.shape[1])

def is_inside_point_cnt(point, cnt):
    # ugly place. point here is (row, col)
    # but in the contour points points are (col, row)
    point = (point[1], point[0])
    return cv2.pointPolygonTest(cnt, point, False) >= 0

def skel_cv(img):
    '''
    img: (row,col) numpy.ndarray, with dtype of np.uint8 PIC_TYPE,
         with element only 0 and 255, 255 for foreground, 0 for background
         img is a mask
    return skel: (row,col) numpy.ndarray, with dtype of bool
    '''
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel > 0

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

# algo0: morphology.medial_axis
# algo1: opencv
# algo2: morphology.skeletonize


def get_maps_algo0(im, cnts):
    skels_points = []
    radius_dict = {}
    score_dict = {}
    curvature_dict = {}
    theta_dict = {}
    mask_fills = []

    for cnt in cnts:
        cnt = np.squeeze(cnt)
        mask_zero = np.zeros(im.shape[:2], dtype = np.uint8)
        mask_fill = cv2.fillPoly(mask_zero, pts = [cnt], color=(255))
        mask_fills.append(np.sign(mask_fill.copy()).astype(MAP_TYPE))
        skel_mask, dist_mask = morphology.medial_axis(mask_fill, return_distance=True)
        skel_points = np.argwhere(skel_mask == True)
        skel_points = [tuple(point) for point in skel_points]
        skel_points = find_mid_line(skel_points)
        [skels_points.append(point) for point in skel_points]
        dist_mask = dist_mask.astype(MAP_TYPE)

        # get a belt and get the thickness on the skel points
        thickness_dict = {}
        belt = set()

        for point in skel_points:
            thickness = int(THICKNESS*dist_mask[point[0],point[1]])
            thickness_dict[point] = thickness
            for i in range(-thickness, thickness+1):
                for j in range(-thickness, thickness+1):
                    candidate = (point[0]+i, point[1]+j)
                    if is_validate_point(im, candidate):
                        belt.add(candidate)

        for point in skel_points:
            fit_points = []
            width = int(NEIGHBOR*dist_mask[point[0],point[1]])
            for fit_point in skel_points:
                if point[0]-width <= fit_point[0] <= point[0]+width and \
                   point[1]-width <= fit_point[1] <= point[1]+width:
                    fit_points.append(fit_point)
            if len(fit_points) >= 2:
                theta = get_theta(fit_points)
            else:
                theta = 0.0
            theta_dict[point] = theta

        # score map
        for point in belt:
            score_dict[point] = 1.0

        # raius map & theta map
        for point in belt:
            dist = []
            for skel_point in skel_points:
                thickness = thickness_dict[skel_point]
                if skel_point[0]-thickness <= point[0] <= skel_point[0]+thickness and \
                   skel_point[1]-thickness <= point[1] <= skel_point[1]+thickness:
                    dist.append((get_l2_dist(point, skel_point), skel_point))
            dist = sorted(dist)

            radius_dict[point] = dist_mask[dist[0][1][0], dist[0][1][1]]
            theta_dict[point] = theta_dict[dist[0][1][0], dist[0][1][1]]

        # curvature map
        if len(cnt) == 4:
            for point in belt:
                curvature_dict[tuple(point)] = 0.0
        else:
            for point in belt:
                curvature_dict[tuple(point)] = 0.0
    return skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills

def get_maps_algo1(im, cnts):
    '''
    skel = skel_cv
    skel_sub = medial_axis + find_mid_line_with_dist
    belt的生成用的是skel_cv+膨胀的方式
    belt上的score是1.0，curvature是0.0，radius是dist_mask上的，theta是用skel_sub上的点fit出来
    '''
    skels_points = []
    radius_dict = {}
    score_dict = {}
    curvature_dict = {}
    theta_dict = {}
    mask_fills = []

    for cnt in cnts:
        cnt = np.squeeze(cnt)
        mask_zero = np.zeros(im.shape[:2], dtype = np.uint8)
        mask_fill = cv2.fillPoly(mask_zero, pts = [cnt], color=(255))
        mask_fills.append(np.sign(mask_fill.copy()).astype(MAP_TYPE))
        skel_mask_sub, dist_mask = morphology.medial_axis(mask_fill, return_distance=True)
        skel_points_sub = np.argwhere(skel_mask_sub == True)
        skel_points_sub = [tuple(point) for point in skel_points_sub]
        skel_points_sub = find_mid_line_with_dist(skel_points_sub, dist_mask)

        mask_zero = np.zeros(im.shape[:2], dtype = np.uint8)
        mask_fill = cv2.fillPoly(mask_zero, pts = [cnt], color=(255))
        skel_mask = skel_cv(mask_fill)
        skel_points = np.argwhere(skel_mask == True)
        skel_points = [tuple(point) for point in skel_points]
        [skels_points.append(point) for point in skel_points]

        dist_mask = dist_mask.astype(MAP_TYPE)

        # get a belt and get the thickness on the skel points
        belt = set()

        for point in skel_points:
            thickness = int(THICKNESS*dist_mask[point[0],point[1]])
            for i in range(-thickness, thickness+1):
                for j in range(-thickness, thickness+1):
                    candidate = (point[0]+i, point[1]+j)
                    if is_validate_point(im, candidate) and \
                       is_inside_point_cnt(candidate, cnt):
                        belt.add(candidate)

        # get the theta on skel_points_sub
        for point in skel_points_sub:
            fit_points = []
            increment = 0.0
            while len(fit_points) < 2:
                width = int(NEIGHBOR*dist_mask[point[0],point[1]]*(1+increment))
                for fit_point in skel_points_sub:
                    if point[0]-width <= fit_point[0] <= point[0]+width and \
                       point[1]-width <= fit_point[1] <= point[1]+width:
                        fit_points.append(fit_point)
                increment += 0.1
            theta = get_theta(fit_points)
            theta_dict[point] = theta

        # score map
        for point in belt:
            score_dict[point] = 1.0

        # raius map
        for point in belt:
            radius_dict[point] = dist_mask[point[0], point[1]]

        # theta map
        for point in belt:
            dist = []
            increment = 0.0
            while len(dist) < 1:
                thickness = int(dist_mask[point[0],point[1]])
                if thickness == 0: thickness = 1
                thickness = int(thickness*(1+increment))
                for skel_point in skel_points_sub:
                    if skel_point[0]-thickness <= point[0] <= skel_point[0]+thickness and \
                       skel_point[1]-thickness <= point[1] <= skel_point[1]+thickness:
                        dist.append((get_l2_dist(point, skel_point), skel_point))
                increment += 0.1
            dist = sorted(dist)
            theta_dict[point] = theta_dict[dist[0][1][0], dist[0][1][1]]

        # curvature map
        if len(cnt) == 4:
            for point in belt:
                curvature_dict[tuple(point)] = 0.0
        else:
            for point in belt:
                curvature_dict[tuple(point)] = 0.0
    return skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills

def get_maps_algo2(im, cnts):
    '''
    algo2: 将dist mask往里面缩小，score为1，radius是distmask，theta为0，curvature为0
    '''
    skels_points = []
    radius_dict = {}
    score_dict = {}
    curvature_dict = {}
    theta_dict = {}
    mask_fills = []

    for cnt in cnts:
        cnt = np.squeeze(cnt)
        mask_zero = np.zeros(im.shape[:2], dtype = np.uint8)
        mask_fill = cv2.fillPoly(mask_zero, pts = [cnt], color=(255))
        mask_fills.append(np.sign(mask_fill.copy()).astype(MAP_TYPE))
        skel_mask, dist_mask = morphology.medial_axis(mask_fill, return_distance=True)

        dist_mask = dist_mask.astype(MAP_TYPE)

        pixel_num = np.sum(np.sign(dist_mask))
        belt = set()
        for index in np.argsort(dist_mask, axis=None)[-int(pixel_num*FIND_MID):]:
            belt.add((index//dist_mask.shape[0], index%dist_mask.shape[0]))
            skels_points.append((index // dist_mask.shape[0], index % dist_mask.shape[0]))
        # score map
        for point in belt:
            score_dict[point] = 1.0
        # radius map
        for point in belt:
            radius_dict[point] = dist_mask[point[0],point[1]]
    return skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills

def find_mid_line_and_radius(points_list,sampling_rate=500):
    """
    give a polygon depicted by a list of points (in order), return the point list of the center line
    :param points_list: List[Point(x,y)]
           sampling_rate: how many points the sampling takes between the two 'ends'
    :return: List[Point(x,y)]
    """
    points_list = [tuple(point) for point in points_list]
    def length(p1,p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def sampling(p1,p2,sampling_rate):
        x = np.linspace(p1[0], p2[0], sampling_rate)
        y = np.linspace(p1[1], p2[1], sampling_rate)
        return [(x[i],y[i]) for i in range(x.shape[0])]

    def neg_cosine(p1,p2,p3,p4):
        vector_1 = (p2[0]-p1[0],p2[1]-p1[1])
        vector_2 = (p4[0] - p3[0], p4[1] - p3[1])
        return (vector_1[1]*vector_2[1]+vector_1[0]*vector_2[0])/(length(vector_2,(0,0))*length(vector_1,(0,0)))

    def find_center_line_3_point(points_list):
        #brute force
        d1=length(points_list[0],points_list[1])
        d2 = length(points_list[1], points_list[2])
        d3 = length(points_list[2], points_list[0])

        if d1<min(d2,d3):
            sample_result = sampling(points_list[2],((points_list[0][0]+points_list[1][0])/2,(points_list[0][1]+points_list[1][1])/2),sampling_rate=sampling_rate)
        if d2<min(d3,d1):
            sample_result = sampling(points_list[0],((points_list[1][0]+points_list[2][0])/2,(points_list[2][1]+points_list[1][1])/2),sampling_rate=sampling_rate)
        if d3<min(d2,d1):
            sample_result = sampling(points_list[1],((points_list[0][0]+points_list[2][0])/2,(points_list[0][1]+points_list[2][1])/2),sampling_rate=sampling_rate)
        raise NotImplementedError('3 points are not supported')

    def find_center_line_4_point(points_list):
        # brute force
        d1=length(points_list[0],points_list[1])
        d2 = length(points_list[1], points_list[2])
        d3 = length(points_list[2], points_list[3])
        d4 = length(points_list[3], points_list[0])
        radius_dict = {}
        if d1+d3<d2+d4:
            sample_result = sampling(((points_list[0][0]+points_list[1][0])/2,(points_list[0][1]+points_list[1][1])/2),
                                     ((points_list[2][0]+points_list[3][0])/2,(points_list[2][1]+points_list[3][1])/2),
                                     sampling_rate=sampling_rate)
            crop_length1 = d1/2
            crop_length2 = d3/2
            new = []
            for point in sample_result:
                if length(point, sample_result[0]) >= crop_length1 and \
                   length(point, sample_result[-1]) >= crop_length2:
                    new.append(point)
            sample_result = new
            sample_result = [(int(round(point[0])),int(round(point[1]))) for point in sample_result]
            for point in sample_result:
                radius_dict[point] = (d1+d3)/2
        else:
            sample_result = sampling(((points_list[2][0] + points_list[1][0]) / 2,
                                      (points_list[2][1] + points_list[1][1]) / 2),
                                     ((points_list[0][0] + points_list[3][0]) / 2,
                                      (points_list[0][1] + points_list[3][1]) / 2),
                                     sampling_rate=sampling_rate)
            crop_length1 = d2/2
            crop_length2 = d4/2
            new = []
            for point in sample_result:
                if length(point, sample_result[0]) >= crop_length1 and \
                   length(point, sample_result[-1]) >= crop_length2:
                    new.append(point)
            sample_result = new
            sample_result = [(int(round(point[0])),int(round(point[1]))) for point in sample_result]
            for point in sample_result:
                radius_dict[point] = (d1+d3)/2
        return sample_result, radius_dict

    if len(points_list)==3:
        return find_center_line_3_point(points_list)
    elif len(points_list)==4:
        return find_center_line_4_point(points_list)

    circular_points_list=points_list+points_list[:3]

    cosine_list=[(circular_points_list[i+1],circular_points_list[i+2],
                  neg_cosine(circular_points_list[i+0],
                             circular_points_list[i+1],
                             circular_points_list[i+2],
                             circular_points_list[i+3]),i)
                 for i in range(len(points_list))]

    best=None#(i,j),i<j

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

    point_list_one = []
    point_list_two = []
    total_len_one=sum([length(line_one[i],line_one[i+1]) for i in range(len(line_one)-1)])
    total_len_two = sum([length(line_two[i], line_two[i + 1]) for i in range(len(line_two) - 1)])
    for p in range(len(line_one)-1):
        point_list_one+=sampling(line_one[p+0],line_one[p+1],math.floor(length(line_one[p+0],line_one[p+1])*sampling_rate/total_len_one))

    for p in range(len(line_two) - 1):
        point_list_two+=sampling(line_two[p+0],line_two[p+1],math.floor(length(line_two[p+0],line_two[p+1])*sampling_rate/total_len_two))

    if len(point_list_one)!=len(point_list_two):
        if len(point_list_one)<len(point_list_two):
            point_list_one,point_list_two=point_list_two,point_list_one #by default, one > two
        while len(point_list_one)!=len(point_list_two):
            p=int(random()*len(point_list_one))
            point_list_one.pop(p)

    center_line = []
    radius_dict = {}
    for i in range(len(point_list_one)):
        x1, y1 = point_list_one[i][0], point_list_one[i][1]
        x2, y2 = point_list_two[i][0], point_list_two[i][1]
        x, y = round((x1+x2)/2), round((y1+y2)/2)
        center_line.append((int(round((x1+x2)/2)), int(round((y1+y2)/2))))
        radius_dict[(x,y)] = round(length((x1,y1),(x2,y2))/2)
    temp = []
    temp_dict = {}
    for i in [0, -1]:
        crop_length = radius_dict[center_line[i]]
        for point in center_line:
            if length(point, center_line[i]) >= crop_length:
                temp.append(point)
                temp_dict[point] = radius_dict[point]
    center_line = temp
    radius_dict = temp_dict
    return center_line, radius_dict

def get_maps_algo3(im, cnts):
    skels_points = []
    radius_dict = {}
    score_dict = {}
    curvature_dict = {}
    theta_dict = {}
    mask_fills = []

    for cnt in cnts:
        cnt = np.squeeze(cnt)
        cnt_ = [(point[1],point[0]) for point in cnt]
        skel_points, radius_dict = find_mid_line_and_radius(cnt_, sampling_rate=500)
        [skels_points.append(point) for point in skel_points]
        mask_zero = np.zeros(im.shape[:2], dtype = np.uint8)
        mask_fill = cv2.fillPoly(mask_zero, pts = [cnt], color=(255))
        mask_fills.append(np.sign(mask_fill.copy()).astype(MAP_TYPE))

        # get theta for skel_points
        for point in skel_points:
            width = int(NEIGHBOR*radius_dict[point])
            fit_points = []
            for fit_point in skel_points:
                if point[0] - width <= fit_point[0] <= point[0] + width and \
                        point[1] - width <= fit_point[1] <= point[1] + width:
                    fit_points.append(fit_point)
            theta = get_theta(fit_points)
            theta_dict[point] =theta

        # get belt
        belt = set()
        for point in skel_points:
            thickness = int(THICKNESS*radius_dict[point])
            for i in range(-thickness, thickness+1):
                for j in range(-thickness, thickness+1):
                    candidate = (point[0]+i, point[1]+j)
                    if is_validate_point(im, candidate) and \
                       is_inside_point_cnt(candidate, cnt):
                        belt.add(candidate)
        # score map
        for point in belt:
            score_dict[point] = 1.0

        # theta, raidus map
        for point in belt:
            dist = []
            increment = 0.0
            while len(dist) < 1:
                for skel_point in skel_points:
                    thickness = int(radius_dict[skel_point]*(1+increment))
                    if skel_point[0]-thickness <= point[0] <= skel_point[0]+thickness and \
                       skel_point[1]-thickness <= point[1] <= skel_point[1]+thickness:
                        dist.append((get_l2_dist(point, skel_point), skel_point))
                increment += 0.1
            dist = sorted(dist)
            theta_dict[point] = theta_dict[dist[0][1][0], dist[0][1][1]]
            radius_dict[point] = radius_dict[dist[0][1][0], dist[0][1][1]]

        # curvature map
        if len(cnt) == 4:
            for point in belt:
                curvature_dict[tuple(point)] = 0.0
        else:
            for point in belt:
                curvature_dict[tuple(point)] = 0.0

    return skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills

def get_maps(im, cnts, algo = 3):
    if algo == 0:
        skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills = \
            get_maps_algo0(im,cnts)
    elif algo == 1:
        skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills = \
            get_maps_algo1(im,cnts)
    elif algo ==2:
        skels_points, radius_dict, score_dict, curvature_dict, theta_dict, mask_fills = \
            get_maps_algo2(im,cnts)
    elif algo == 3:
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



