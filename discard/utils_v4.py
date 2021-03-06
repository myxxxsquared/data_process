import numpy as np
import cv2
import math
from random import random
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)


def get_l2_dist(point1, point2):
    '''
    :param point1: tuple (x, y) int or float
    :param point2: tuple (x, y)
    :return: float
    '''
    return float(((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5)


def get_l1_dist(point1, point2):
    '''
    :param point1: tuple (x, y) int or float
    :param point2: tuple (x, y)
    :return: float
    '''
    return float(abs(point1[0]-point2[0])+abs(point1[1]-point2[1]))


def get_cos(point, point1, point2):
    '''
    :param point: tuple (x, y) int or float
    :param point1: tuple (x, y) int or float
    :param point2: tuple (x, y) int or float
    :return: float
    '''
    # point is pivot
    vec1 = (point[0]-point1[0], point[1]-point1[1])
    vec2 = (point[0]-point2[0], point[1]-point2[1])
    return float((vec1[0]*vec2[0]+vec1[1]*vec2[1])/(get_l2_dist(point,point1)*get_l2_dist(point,point2)))


def get_shortest_dist(point, point1, point2):
    '''
    :param point: tuple (x, y) int or float
    :param point1: tuple (x, y) int or float
    :param point2: tuple (x, y) int or float
    :return: float
    '''
    dist1 = get_l2_dist(point, point1)
    dist2 = get_l2_dist(point, point2)
    if get_cos(point1, point, point2) < 0 or get_cos(point2, point, point1) < 0:
        return min(dist1, dist2)
    else:
        return dist1 * math.sqrt(1-get_cos(point1, point, point2)**2)


def get_theta(points_list):
    '''
    :param points_list: list(tuple), tuple (x, y) int or float
    :return: float
    '''
    if len(points_list) < 2:
        raise AttributeError('get_theta need at least 2 points')
    xs, ys = [], []
    for (x,y) in points_list:
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    m,b = np.polyfit(xs, ys, 1)
    theta = np.arctan(m)
    return theta


def get_radius(point, cnt):
    '''
    :param point: (x, y), int or float, x needs to be row, y needs to be col
    :param cnt: cnt: numpy.ndarray, shape (n, 1, 2) or (n, 2), dtype float32 or int32, point order (col, row)
    :return:
    '''
    cnt = np.squeeze(cnt)
    # change the order of the cnt
    cnt_changed = [(point[1], point[0]) for point in cnt]
    cnt_changed = cnt_changed + [cnt_changed[0]]
    dist_list = []
    for i in range(len(cnt_changed)-1):
        dist_list.append(get_shortest_dist(point, cnt_changed[i], cnt_changed[i+1]))
    return max(dist_list)


def is_validate_cnts(im, cnts):
    '''
    :param im: numpy.ndarray, shape (row, col, 3), dtype uint 8
    :param cnts: list(numpy.ndarray), shape (n, 1, 2) or (n, 2), dtype float32 or int32, point order (col, row)
    :return: bool
    '''
    cols, rows = [], []
    for cnt in cnts:
        cnt = np.squeeze(cnt)
        for i in range(len(cnt)):
            cols.append(cnt[i][0])
            rows.append(cnt[i][1])
    col_max = max(cols)
    row_max = max(rows)
    im_row, im_col = im.shape[0]-1, im.shape[1]-1
    flag = True
    if im_row < row_max: flag = False
    if im_col < col_max: flag = False
    return flag


def is_validate_point(im, point):
    '''
    :param im: numpy.ndarray, shape (row, col, 3), dtype uint 8
    :param point: (x, y) int or float , x needs to be row, y needs to be col
    :return: bool
    '''
    row, col = im.shape[:2]
    return (point[0] < row) and (point[1] < col)


def is_inside_point_cnt(point, cnt):
    '''
    :param point: (x, y) int or float, where x is row, y is col
    :param cnt: numpy.ndarray, shape (n, 1, 2), dtype int32 or float32, point order (col, row)
    :return: bool
    '''
    cnt = np.array(cnt, np.float32)
    point = (point[1], point[0])
    return cv2.pointPolygonTest(cnt, point, False) >= 0


def validate(im, cnts):
    '''
    :param im: numpy.ndarray, shape (row, col, 3), dtype uint 8
    :param cnts: list(numpy.ndarray), shape (n, 1, 2) or (n,2), dtype int32 or float32, point order (col, row)
    :return:
        im: numpy.ndarray, shape (row, col, 3), dtype uint 8
        cnts: list(numpy.ndarray), shape (n, 1, 2), dtype float32, point order (col, row)
    '''
    cols, rows = [], []
    for cnt in cnts:
        cnt = np.squeeze(cnt)
        for i in range(len(cnt)):
            cols.append(cnt[i][0])
            rows.append(cnt[i][1])
    col_max = int(max(cols))
    row_max = int(max(rows))
    im_row, im_col = im.shape[0]-1, im.shape[1]-1
    if im_row < row_max:
        temp = np.zeros([row_max-im_row, im.shape[1], im.shape[2]])
        im = np.concatenate((im, temp), 0)
    if im_col < col_max:
        temp = np.zeros([im.shape[0], col_max-im_col, im.shape[2]])
        im = np.concatenate((im, temp), 1)
    cnts = [np.array(np.reshape(cnt, (-1,1,2)), np.float32) for cnt in cnts]
    return im, cnts


def resize(im, cnts, row, col):
    '''
    :param im: numpy.ndarray, shape (row, col, 3), dtype uint 8
    :param cnts: list(numpy.ndarray), shape (n, 1, 2), dtype float32, point order (col, row)
    :param row: int
    :param col: int
    :return:
        im: numpy.ndarray, shape (row, col, 3), dtype uint 8
        cnts: list(numpy.ndarray), shape (n, 1, 2), dtype float32, point order (col, row)
    '''
    im_row, im_col = im.shape[0], im.shape[1]
    im_ = cv2.resize(im, (row, col), interpolation = cv2.INTER_CUBIC)
    cnts_ = []
    for cnt in cnts:
        cnt = np.reshape(cnt, (-1,1,2))
        temp = np.zeros_like(cnt, np.float32)
        for i in range(len(cnt)):
            temp[i][0][0] = cnt[i][0][0]*col/im_col
            temp[i][0][1] = cnt[i][0][1]*row/im_row
        cnts_.append(temp)
    cnts_ = [np.array(np.reshape(cnt, (-1,1,2)), np.float32) for cnt in cnts_]
    return im_, cnts_


def sampling(p1,p2,sampling_num):
    '''
    :param p1: point(x,y) int or float
    :param p2: point(x,y)
    :param sampling_num:
    :return: points_list, list(point), float
    '''
    x = np.linspace(p1[0], p2[0], sampling_num)
    y = np.linspace(p1[1], p2[1], sampling_num)
    return [(x[i],y[i]) for i in range(x.shape[0])]


def get_cos_with_4(p1,p2,p3,p4):
    '''
    :param p1: point(x,y) int or float
    :param p2: point(x,y) int or float
    :param p3: point(x,y) int or float
    :param p4: point(x,y) int or float
    :return: float
    '''
    vector_1 = (p2[0]-p1[0],p2[1]-p1[1])
    vector_2 = (p4[0]-p3[0], p4[1]-p3[1])
    if get_l2_dist(vector_2,(0,0))*get_l2_dist(vector_1,(0,0)) == 0.0:
        return 0.0
    return (vector_1[1]*vector_2[1]+vector_1[0]*vector_2[0])/(get_l2_dist(vector_2,(0,0))*get_l2_dist(vector_1,(0,0)))


def find_mid_line_with_radius_theta(points_list, crop_skel, neighbor, sampling_num=500):
    '''
    :param points_list: list(tuple), tuple (x, y)
    :param sampling_num: int. the number to sample on one line for a cnt
    :return: center_line, radius_dict, theta_dict
        center_line: list(tuple), tuple (x, y), x y are int
        radius_dict: dict, key is tuple (x, y), value is float
        theta_dict: dict, key is tuple (x, y), value is float
    '''

    points_list = [tuple(point) for point in points_list]

    if len(points_list) > 4:
        circular_points_list=points_list+points_list[:3]
        cosine_list=[(circular_points_list[i+1],circular_points_list[i+2],
                      get_cos_with_4(circular_points_list[i+0],
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

    center_line_set = set()
    center_line = []
    radius_dict = {}
    theta_dict = {}

    for i in range(len(point_list_one)):
        x1, y1 = point_list_one[i][0], point_list_one[i][1]
        x2, y2 = point_list_two[i][0], point_list_two[i][1]
        x, y = int(round((x1+x2)/2)), int(round((y1+y2)/2))
        if not((x,y) in center_line_set):
            center_line.append((x,y))
        center_line_set.add((x,y))
        radius_dict[(x,y)] = get_l2_dist((x1,y1),(x2,y2))/2
    if len(points_list) == 4:
        theta = get_theta([center_line[0]]+center_line[::30]+[center_line[-1]])
        for point in center_line:
            theta_dict[point] = theta
    else:
        neighbor = int(neighbor)
        for p,point in enumerate(center_line):
            theta = get_theta(list(set(center_line[max(p-neighbor,0):p+neighbor])))
            theta_dict[point] = theta

    temp = []
    temp_radius_dict = {}
    temp_theta_dict = {}
    crop_length1 = radius_dict[center_line[0]]
    crop_length2 = radius_dict[center_line[-1]]
    decrease = 0.0
    while len(temp) <= 1:
        for point in center_line:
            crop_length1 = crop_length1*(1-decrease)
            crop_length2 = crop_length2*(1-decrease)
            if get_l2_dist(point, center_line[0]) >= crop_length1*crop_skel and \
               get_l2_dist(point, center_line[-1]) >= crop_length2 * crop_skel:
                temp.append(point)
                temp_radius_dict[point] = radius_dict[point]
                temp_theta_dict[point] = theta_dict[point]
        decrease += 0.01
    center_line = temp
    radius_dict = temp_radius_dict
    theta_dict = temp_theta_dict
    return center_line, radius_dict, theta_dict


def get_maps_textbox(im, cnts, thickness,crop_skel, neighbor):
    '''
    :param im: numpy.ndarray, shape (row, col, 3), dtype uint 8
    :param cnts: list(numpy.ndarray), shape (n, 1, 2), dtype int32 or float32, point order (col, row)
    :return: skels_points, radius_dict, cos_theta_dict, sin_theta_dict, theta_dict, mask_fills
        skels_points: list(tuple), tuple (x, y), x y are int
        radius_dict: dict, key is tuple (x, y), value is float
        score_dict: dict, key is tuple (x, y), value is bool
        cos_theta_dict: dict, key is tuple (x, y), value is float
        sin_theta_dict: dict, key is tuple (x, y), value is float
        mask_fills: list(numpy.ndarray), numpy.ndarray shape (row, col, 1), dtype bool
    '''

    skels_points = []
    radius_dict = {}
    score_dict = {}
    theta_dict = {}
    cos_theta_dict = {}
    sin_theta_dict = {}
    mask_fills = []
    for cnt in cnts:
        cnt = np.squeeze(cnt)
        # change cordinate from cnt to point_list
        point_list = [(point[1],point[0]) for point in cnt]
        skel_points, radius_dict_cnt, theta_dict_cnt = \
            find_mid_line_with_radius_theta(point_list, crop_skel, neighbor, sampling_num=500)

        for point, radius in radius_dict_cnt.items():
            radius_dict[point] = radius
        for point, theta in theta_dict_cnt.items():
            theta_dict[point] = theta
        for point in skel_points:skels_points.append(point)

        mask_fill = np.zeros(im.shape[:2], dtype = np.uint8)
        cnt_ = np.array(cnt, np.int32)
        mask_fill = cv2.fillPoly(mask_fill, pts = [cnt_], color=(255))
        mask_fills.append(mask_fill.copy().astype(np.bool))

        # get belt
        belt = set()
        connect_dict = {}
        for point in skel_points:
            r = int(thickness*radius_dict[point])
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    candidate = (point[0]+i, point[1]+j)
                    if is_validate_point(im, candidate):
                        belt.add(candidate)
                        if candidate not in connect_dict:
                            connect_dict[candidate] = []
                        connect_dict[candidate].append(point)

        # score map
        for point in belt:
            score_dict[point] = True

        # theta, raidus map
        for point in belt:
            min_dist = 1e8
            min_dist_point = None
            for skel_point in connect_dict[point]:
                dist = get_l2_dist(point, skel_point)
                if dist < min_dist:
                    min_dist_point = skel_point
                    min_dist = dist
            cos_theta_dict[point] = math.cos(theta_dict[min_dist_point[0], min_dist_point[1]])
            sin_theta_dict[point] = math.sin(theta_dict[min_dist_point[0], min_dist_point[1]])
            radius_dict[point] = radius_dict[min_dist_point[0], min_dist_point[1]]-min_dist

    return skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills


def find_mid_line_with_radius_theta_char(char_cnt_per_text, sampling_num=500):
    '''
    :param char_cnt_per_text: list(tuple(point, char_cnt)); point: (x, y) int;
                            char_cnt: np.ndarray(4,2) or (4,1,2)
    :param crop_skel:
    :param neighbor:
    :param sampling_num:
    :return:
        center_line: list(tuple), tuple (x, y), x y are int
        radius_dict: dict, key is tuple (x, y), value is float
        theta_dict: dict, key is tuple (x, y), value is float

    '''
    radius_dict = {}
    theta_dict = {}

    skel_points = set()
    len_ = len(char_cnt_per_text)

    for point, char_cnt in char_cnt_per_text:
        radius_dict[point] = get_radius(point, char_cnt)

    for i in range(len_):
        skel_points.add(char_cnt_per_text[i][0])
    for i in range(len_-1):
        point1, point2 = char_cnt_per_text[i][0], char_cnt_per_text[i+1][0]
        sample_points = sampling(point1, point2, sampling_num)
        sample_radius = np.linspace(radius_dict[point1], radius_dict[point2], sampling_num)
        if i == 0:
            theta = get_theta([point1, point2])
        else:
            theta = 0.5*(get_theta([point1, point2])+get_theta([point1, char_cnt_per_text[i-1][0]]))
        for point, radius in zip(sample_points, sample_radius):
            sample = (int(round(point[0])), int(round(point[1])))
            skel_points.add(sample)
            radius_dict[sample] = radius
            theta_dict[sample] = theta
    if len_ == 1:
        theta_dict[char_cnt_per_text[0][0]] = math.pi/2
    for i in range(len_):
        assert char_cnt_per_text[i][0] in radius_dict
        assert char_cnt_per_text[i][0] in theta_dict
    return skel_points, radius_dict, theta_dict


def get_center_point(cnt):
    '''
    :param cnt: list(tuple(col,row))
    :return: tuple(x, y) x, y are int, x is row, y is col
    '''
    cnt = np.squeeze(cnt)
    xs, ys = [], []
    for point in cnt:
        xs.append(point[1])
        ys.append(point[0])
    return int(round(sum(xs)/len(xs))), int(round(sum(ys)/len(ys)))


def reorder(char_cnt_per_text):
    '''
    :param char_cnt_per_text: list(tuple(point, char_cnt)); point: (x, y) int;
            char_cnt: np.ndarray(4,2) or(4,1,2)
    :return: char_cnt_per_text, same as the input
    '''
    # assert char_cnt_per_text[0][1].shape == (4, 2), char_cnt_per_text[0]
    len_ = len(char_cnt_per_text)
    if len_ == 0:
        raise AttributeError('did not find char in text')
    if len_ == 1:
        return char_cnt_per_text

    info = np.zeros((len_, len_))
    for i in range(len_):
        for j in range(len_):
            dist = get_l2_dist(char_cnt_per_text[i][0], char_cnt_per_text[j][0])
            info[i, j] = info[j, i] = dist
    tree = set()
    tree.add(0)
    remain = set(range(1,len_))
    path = []

    while len(tree) < len_:
        dist_list = []
        for start in tree:
            for end in remain:
                dist_list.append((info[start,end], start, end))
        start,end = sorted(dist_list)[0][1:]
        path.append((start, end))
        tree.add(end)
        remain.remove(end)

    # assert that there is only one path in the tree
    count = [0 for _ in range(len_)]
    for start, end in path:
        count[start]+=1
        count[end]+=1
    if max(count) > 2:
        return None

    deque = []
    start, end = path[0]
    path.pop(0)
    deque.append(start)
    deque.append(end)

    for _ in range(len_):
        for i in range(len(path)):
            if path[i][0] == deque[0]:
                deque.insert(0, path[i][1])
            elif path[i][0] == deque[-1]:
                deque.append(path[i][1])

    assert len(deque) == len_
    new = []
    for index in deque:
        new.append(char_cnt_per_text[index])
    return new


def reconstruct(skel_points, radius_dict_cnt, row, col):
    '''
    :param skel_points: list(tuple(x,y))
    :param radius_dict_cnt:
    :return:
        mask_fill numpy.ndarray shape (row, col, 1), dtype bool
    '''
    # denote that: when changing from point_list to hull or cnt
    # we need to change the coordination
    zeros = np.zeros((row, col), np.uint8)
    for point in skel_points:
        radius = radius_dict_cnt[point]
        radius = int(round((radius)))
        zeros = cv2.circle(zeros, (point[1], point[0]), radius, (255), -1)
    _,cnts,_ = cv2.findContours(zeros, 1, 2)
    mask_fill = np.zeros((row, col), np.uint8)
    cnts = [np.array(cnt, np.int32) for cnt in cnts]
    mask_fill = cv2.fillPoly(mask_fill, cnts, (255)).astype(np.bool)
    return mask_fill


def get_maps_charbox(im, cnts, thickness, crop_skel, neighbor):
    '''
    :param im: numpy.ndarray, shape (row, col, 3), dtype uint 8
    :param cnts: list(list(numpy.ndarray)), shape (n, 1, 2), dtype int32, point order (col, row)
                [text_cnts, char_cnts]
    :param thickness: float
    :param neighbor: float
    :param crop_skel: float
    :return:
        skels_points: list(tuple), tuple (x, y), x y are int
        radius_dict: dict, key is tuple (x, y), value is float
        score_dict: dict, key is tuple (x, y), value is bool
        cos_theta_dict: dict, key is tuple (x, y), value is float
        sin_theta_dict: dict, key is tuple (x, y), value is float
        mask_fills: list(numpy.ndarray), numpy.ndarray shape (row, col, 1), dtype bool

    '''
    skels_points = []
    radius_dict = {}
    score_dict = {}
    theta_dict = {}
    cos_theta_dict = {}
    sin_theta_dict = {}
    mask_fills = []

    char_cnts, text_cnts = cnts

    while len(text_cnts) != 0:
        text_cnt = text_cnts.pop(0)
        char_cnt_per_text = []

        char_cnt_index = []
        for index, char_cnt in enumerate(char_cnts):
            center_point = get_center_point(char_cnt)
            if is_inside_point_cnt(center_point, text_cnt):
                char_cnt_per_text.append((center_point, char_cnt))
                char_cnt_index.append(index)

        char_cnt_per_text = reorder(char_cnt_per_text)

        # text_cnts are recalled only char_cnt_per_text returns None
        # in case that text_cnts overlap with each other
        if char_cnt_per_text is None:
            text_cnts.append(text_cnt)
        else:
            # delete claimed char_cnts
            char_cnts_temp = []
            for i in range(len(char_cnts)):
                if i not in char_cnt_index:
                    char_cnts_temp.append(char_cnts[i])
            char_cnts = char_cnts_temp

            # it char_cnt is a text_cnt itself,
            # treat it as a text_cnt and call the function for text_cnt
            if len(char_cnt_per_text) == 1:
                point_list = [(point[1], point[0]) for point in char_cnt_per_text[0][1]]
                skel_points, radius_dict_cnt, theta_dict_cnt = \
                    find_mid_line_with_radius_theta(point_list, crop_skel, neighbor)
            else:
                skel_points, radius_dict_cnt, theta_dict_cnt = \
                    find_mid_line_with_radius_theta_char(char_cnt_per_text, sampling_num=500)

            for point, radius in radius_dict_cnt.items():
                radius_dict[point] = radius
            for point, theta in theta_dict_cnt.items():
                theta_dict[point] = theta
            [skels_points.append(point) for point in skel_points]

            mask_fill = reconstruct(skel_points, radius_dict_cnt, im.shape[0], im.shape[1])
            mask_fills.append(mask_fill.astype(np.bool))

            # get belt
            belt = set()
            connect_dict = {}

            for point in skel_points:
                r = int(thickness*radius_dict[point])
                for i in range(-r, r+1):
                    for j in range(-r, r+1):
                        candidate = (point[0]+i, point[1]+j)
                        if is_validate_point(im, candidate):
                            belt.add(candidate)
                            if candidate not in connect_dict:
                                connect_dict[candidate] = []
                            connect_dict[candidate].append(point)

            # score map
            for point in belt:
                score_dict[point] = True

            # theta, raidus map
            for point in belt:
                min_dist = 1e8
                min_dist_point = None
                for skel_point in connect_dict[point]:
                    dist = get_l2_dist(point, skel_point)
                    if dist < min_dist:
                        min_dist_point = skel_point
                        min_dist = dist
                cos_theta_dict[point] = math.cos(theta_dict[min_dist_point[0], min_dist_point[1]])
                sin_theta_dict[point] = math.sin(theta_dict[min_dist_point[0], min_dist_point[1]])
                radius_dict[point] = radius_dict[min_dist_point[0], min_dist_point[1]]-min_dist

    return skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills


def get_maps(im, cnts, is_textbox, thickness, crop_skel, neighbor):
    '''
    :param im: numpy.ndarray, shape (row, col, 3), dtype uint 8
    :param cnts:
           if is_textbox is True:
            cnts: list(numpy.ndarray), shape (n, 1, 2), dtype int32 or float32, point order (col, row) for textbox
           if is_textbox is False:
            cnts: [char_cnts, text_cnts], both char_cnts and text_cnts are the same as the text_cnts above
    :param is_textbox: bool
    :param thickness: float, used to span a belt from center(skeleton) line, used both for (text_cnts) and (char&text cnts)
    :param crop_skel: float, used to crop the head and end of center(skeleton) line, used for (text_cnts)
    :param neighbor: int, used to determine the range for fitting aline to get theta, used both for (text_cnts) and (char&text cnts)
    :return: skels_points: list(tuple) center(skeleton) line, (x, y) int, x is row, y is col
             radius_dict, score_dict, cos_theta_dict, sin_theta_dict: key is tuple(x,y), value is float
             mask_fills: list(np.ndarray) shape(row,col,1), dtype bool
    '''

    if is_textbox:
        cnts = [np.array(cnt, np.float32) for cnt in cnts]

        skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
            get_maps_textbox(im,cnts, thickness, crop_skel, neighbor)
    else:
        char_cnts, text_cnts = cnts
        char_cnts = [np.array(cnt, np.float32) for cnt in char_cnts]
        text_cnts = [np.array(cnt, np.float32) for cnt in text_cnts]
        cnts = [char_cnts, text_cnts]
        skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
            get_maps_charbox(im,cnts, thickness, crop_skel, neighbor)
    return skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills


if __name__ == '__main__':
    ########### test text_cnts ############

    # PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    # import pickle
    #
    # for i in range(9, 10):
    #     res = pickle.load(open(PKL_DIR+'totaltext_train/'+str(i)+'.bin', 'rb'))
    #     print(res['img_name'],
    #           res['contour'],
    #           res['img'])
    #
    #     img_name = res['img_name']
    #     img = res['img']
    #     cnts = res['contour']
    #     is_text_cnts = res['is_text_cnts']
    #
    #     skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
    #         get_maps(img, cnts, is_text_cnts, 0.15, 1.0, 2)
    #     TR = mask_fills[0]
    #     for i in range(1, len(mask_fills)):
    #         TR = np.bitwise_or(TR, mask_fills[i])
    #     TCL = np.zeros(img.shape[:2], np.bool)
    #     for point, _ in score_dict.items():
    #         TCL[point[0], point[1]] = True
    #     radius = np.zeros(img.shape[:2], np.float32)
    #     for point, r in radius_dict.items():
    #         radius[point[0], point[1]] = r
    #     cos_theta = np.zeros(img.shape[:2], np.float32)
    #     for point, c_t in cos_theta_dict.items():
    #         cos_theta[point[0], point[1]] = c_t
    #     sin_theta = np.zeros(img.shape[:2], np.float32)
    #     for point, s_t in sin_theta_dict.items():
    #         sin_theta[point[0], point[1]] = s_t
    #
    #
    #     def save_heatmap(save_name, map):
    #         map = np.array(map, np.float32)
    #         if np.max(map) != 0.0 or np.max(map) != 0:
    #             cv2.imwrite(save_name, (map * 255 / np.max(map)).astype(np.uint8))
    #         else:
    #             cv2.imwrite(save_name, map.astype(np.uint8))
    #     cv2.imwrite(img_name+'.jpg', img)
    #     zeros = np.zeros_like(img)
    #     cnts = [np.array(cnt, np.int32) for cnt in cnts]
    #     zeros = cv2.drawContours(zeros, cnts, -1, (255,255,255), 1)
    #     cv2.imwrite(img_name+'_box.jpg', zeros)
    #     save_heatmap(img_name+'_TR.jpg', TR)
    #     save_heatmap(img_name+'_TCL.jpg', TCL)
    #     save_heatmap(img_name+'_radius.jpg', radius)
    #     save_heatmap(img_name+'_cos_theta.jpg', cos_theta)
    #     save_heatmap(img_name+'_sin_theta.jpg', sin_theta)

    ######## test char_cnts and text_cnts ############
    PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    import pickle

    for i in (662602):
        res = pickle.load(open(PKL_DIR+'synthtext/'+str(i)+'.bin', 'rb'))
        print(res['img_name'],
              res['contour'],
              res['img'])

        img_name = res['img_name']
        img_name = img_name.replace('/', '_')
        img = res['img']
        cnts = res['contour']
        is_text_cnts = res['is_text_cnts']
        cv2.imwrite(img_name+'.jpg', img)
        char_cnts, text_cnts = cnts
        zeros = np.zeros_like(img)
        char_cnts = [np.array(cnt, np.int32) for cnt in char_cnts]
        text_cnts = [np.array(cnt, np.int32) for cnt in text_cnts]
        zeros = cv2.drawContours(zeros, char_cnts, -1, (0,0,255), 1)
        zeros = cv2.drawContours(zeros, text_cnts, -1, (255,255,255), 1)
        cv2.imwrite(img_name+'_box.jpg', zeros)

        skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
            get_maps(img, cnts, is_text_cnts, 0.15, 1.0, 2)
        TR = mask_fills[0]
        for i in range(1, len(mask_fills)):
            TR = np.bitwise_or(TR, mask_fills[i])
        TCL = np.zeros(img.shape[:2], np.bool)
        for point, _ in score_dict.items():
            TCL[point[0], point[1]] = True
        radius = np.zeros(img.shape[:2], np.float32)
        for point, r in radius_dict.items():
            radius[point[0], point[1]] = r
        cos_theta = np.zeros(img.shape[:2], np.float32)
        for point, c_t in cos_theta_dict.items():
            cos_theta[point[0], point[1]] = c_t
        sin_theta = np.zeros(img.shape[:2], np.float32)
        for point, s_t in sin_theta_dict.items():
            sin_theta[point[0], point[1]] = s_t


        def save_heatmap(save_name, map):
            map = np.array(map, np.float32)
            if np.max(map) != 0.0 or np.max(map) != 0:
                cv2.imwrite(save_name, (map * 255 / np.max(map)).astype(np.uint8))
            else:
                cv2.imwrite(save_name, map.astype(np.uint8))

        save_heatmap(img_name+'_TR.jpg', TR)
        save_heatmap(img_name+'_TCL.jpg', TCL)
        save_heatmap(img_name+'_radius.jpg', radius)
        save_heatmap(img_name+'_cos_theta.jpg', cos_theta)
        save_heatmap(img_name+'_sin_theta.jpg', sin_theta)
