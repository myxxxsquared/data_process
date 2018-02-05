import numpy as np
import cv2
from utils import get_maps

def get_l2_dist(point1, point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5

def evaluate(img, cnts, is_text_cnts, maps, is_viz,
             save_name=None, fsk=0.8, tp=0.4, tr=0.8):
    '''
    :param img: ndarrray, np.uint8,
    :param cnts:
        if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2]
        if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
    :param is_text_cnts: bool
    :param maps:
        maps: [TR, TCL, radius, cos_theta, sin_theta], all of them are 2-d array,
        TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
    :param is_viz: bool
    :param save_name
           if is_viz is True, save_name is used to save viz pics
    :return:

    '''
    if not is_text_cnts:
        char_cnts, text_cnts = cnts
        cnts = text_cnts

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
                    x_next = x+direction_x[i]
                    y_next = y+direction_y[j]
                    if x_next < row and y_next < col and \
                            cropped_TCL_for_search[x_next, y_next] == True:
                        queue.append((x_next, y_next))
                        cropped_TCL_for_search[x_next, y_next] = False
        instances.append(instance)

    # for each instance build its bounding box(represented by cnt)
    reconstructed_cnts = []
    for instance in instances:
        zeros = np.zeros((row, col), np.uint8)
        for x, y in instance:
            r = radius[x, y]
            print(r)
            for i in range(-int(r), int(r)+1):
                for j in range(-int(r), int(r)+1):
                    next_x, next_y = x+i, y+j
                    if next_x < row and next_y < col and \
                        get_l2_dist((next_x, next_y), (x, y)) <= r:
                        zeros[next_x, next_y] = 1
        _,cnt,_ = cv2.findContours(zeros, 1, 2)
        if len(cnt) > 1:
            print('more than one cnt')
            for cnt_ in cnt:
                reconstructed_cnts.append(cnt_)
        else:
            reconstructed_cnts.append(cnt)
    reconstructed_cnts = [np.reshape(np.array(reconstructed_cnt, np.float32), (-1, 1, 2)) \
                          for reconstructed_cnt in reconstructed_cnts]


    #TODO
    if is_viz:
        def save_heatmap(save_name, map):
            if np.max(map) != 0.0 or np.max(map) != 0:
                cv2.imwrite(save_name, map.astype(np.uint8) * 255 / np.max(map))
            else:
                cv2.imwrite(save_name, map.astype(np.uint8))
        save_heatmap('cropped_TCL.jpg', cropped_TCL)
        save_heatmap('TR.jpg', TR)
        save_heatmap('TCL.jpg', TCL)
        save_heatmap('radius.jpg', radius)
        cv2.imwrite('img.jpg', img)
        assert save_name is not None
        viz = np.zeros(img.shape,np.uint8)
        cnts = [np.array(cnt, np.int32) for cnt in cnts]
        viz = cv2.drawContours(viz, cnts, -1, (255,255,255), 1)
        reconstructed_cnts = [np.array(cnt, np.int32) for cnt in reconstructed_cnts]
        viz = cv2.drawContours(viz, reconstructed_cnts, -1, (0,0,255), 1)
        cv2.imwrite('box.jpg', viz)

    cnts_num = len(cnts)
    re_cnts_num = len(reconstructed_cnts)

    cnts_mask = []
    re_cnts_mask = []

    for i in range(cnts_num):
        zeros = np.zeros(img.shape[:2], np.uint8)
        cnts_mask.append(cv2.fillPoly(zeros, [cnts[i]], (255)).astype(np.bool))
    for i in range(re_cnts_num):
        zeros = np.zeros(img.shape[:2], np.uint8)
        re_cnts_mask.append(cv2.fillPoly(zeros, [reconstructed_cnts[i]], (255)).astype(np.bool))


    precise = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            precise[i,j] = np.sum(cnts_mask[i]&re_cnts_mask[j])/np.sum(re_cnts_mask[j])

    recall = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            recall[i,j] = np.sum(cnts_mask[i]&re_cnts_mask[j])/np.sum(cnts_mask[i])

    interset = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            interset[i,j] = np.sum(cnts_mask[i]&re_cnts_mask[j])

    one_to_many_score = np.zeros((cnts_num), np.float32)
    for i in range(cnts_num):
        # one to many (splits)
        p_list = []
        index_list =[]
        for index, p in enumerate(precise[i,:]):
            if p >= tp:
                p_list.append(p)
                index_list.append(index)
        r_sum = 0.0
        for index in index_list:
            r_sum += recall[i, index]
        if r_sum >= tr:
            one_to_many_score[i] = fsk

    many_to_one_score = np.zeros((re_cnts_num), np.float32)
    for j in range(re_cnts_num):
        # many to one (merge)
        r_list = []
        index_list = []
        for index, r in enumerate(recall[:, j]):
            if r >= tr:
                r_list.append(r)
                index_list.append(index)
        p_sum = 0.0
        for index in index_list:
            p_sum += precise[index, j]
        if p_sum >= tp:
            many_to_one_score[j] = fsk

    # one to one, overwrite
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            p_ = precise[i,j]
            r_ = recall[i,j]

            # one to one
            if np.sum(interset[i,:])-interset[i,j] == 0.0 and \
               np.sum(interset[:, j]) - interset[i, j] == 0.0 and \
               p_ >= tp and r_ >= tr:
               one_to_many_score[i] = 1.0
               many_to_one_score[j] = 1.0

    final_recall = np.sum(one_to_many_score)/cnts_num
    final_precision = np.sum(many_to_one_score)/re_cnts_num

    return final_precision, final_recall

if __name__ == '__main__':
    EVALUATE_DIR = '/home/rjq/data_cleaned/data_cleaned/evaluate/'
    PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    import pickle

    # ######test char&text cnts##########
    # for i in range(9, 10):
    #     res = pickle.load(open(PKL_DIR+'synthtext/'+str(i)+'.bin', 'rb'))

    ######test text cnts###############
    for i in range(9, 10):
        res = pickle.load(open(PKL_DIR + 'totaltext_train/' + str(i) + '.bin', 'rb'))

        print(res['img_name'],
              res['contour'],
              res['img'],
              res['is_text_cnts'])

        img_name = res['img_name']
        img_name = img_name.replace('/', '_')
        img = res['img']
        cnts = res['contour']
        is_text_cnts = res['is_text_cnts']

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

        maps = [TR, TCL, radius, cos_theta, sin_theta]
        precision, recall = evaluate(img, cnts, is_text_cnts, maps, True, img_name)
        print('precision', precision)
        print('recall', recall)







