import numpy as np
import math, cv2
import argparse
import os
import scipy.io as sio
from utils import resize

RESIZE_GT_ROW = 128
RESIZE_GT_COL = 128

SAVE_DIR = '/home/rjq/data_cleaned/data_cleaned/evaluate/'
INPUT_DIR = '/home/rjq/data_cleaned/data_cleaned/'
TOTALTEXT_DIR = '/home/rjq/data/Total-Text-Dataset/Download/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='input dir')
    parser.add_argument('--output_dir', type=str, default='', help='output dir')
    parser.add_argument('--check_num', type=int, default=10, help ='check num')
    return parser.parse_args()

def check(args):
    input_dir = args.input_dir.rstrip('/')
    args.input_dir = input_dir+'/'
    if args.output_dir == '':
        args.output_dir = args.input_dir
    if not os.path.exists(SAVE_DIR+args.output_dir):
        os.makedirs(SAVE_DIR+args.output_dir)
    return args

def reconstruction(output):
    score=output[:,:,0] # 0/1 mask
    theta=output[:,:,1]
    curvature=output[:,:,2]
    radius=output[:,:,3]
    mask=output[:, :, 4] # 0/1 mask
    masked_score=score*mask # multiplication is faster and logical '&' here.

    def apply_circle(image,x,y,r):
        for i in range(max(0,math.floor(x-r)), min(int(image.shape[0]), math.ceil(x+r))+1):
            for j in range(max(0, math.floor(y - r)), min(int(image.shape[1]), math.ceil(y + r)) + 1):
                if i < image.shape[0] and j < image.shape[1] and (i-x)**2+(y-j)**2<=r**2:
                    image[i,j]=1
        return image

    def find_region(x,y,size=output.shape[:2]):
        instance_map=np.zeros(shape=size,dtype=np.int32)
        queue=[(x,y)]
        direction=((-1,-1),
                   (-1,0),
                   (-1,1),
                   (0,-1),
                   (0,1),
                   (1,-1),
                   (1,0),
                   (1,1))
        masked_score[queue[-1]] = 0
        while len(queue)>0:
            cur_point=queue.pop(0)
            instance_map=apply_circle(instance_map,*cur_point,radius[cur_point])
            for i in range(8):
                x_,y_=cur_point[0]+direction[i][0],cur_point[1]+direction[i][1]
                if x_<0 or y_<0 or x_>=size[0] or y_>=size[1]:
                    continue
                if masked_score[x_,y_]==1.0:
                    queue.append((x_,y_))
                    masked_score[x_,y_] = 0
        return instance_map, np.transpose((np.nonzero(instance_map)[1], np.nonzero(instance_map)[0]))

    instance_list=[]
    instance_points_list=[]
    convex_hulls=[]

    for i in range(masked_score.shape[0]):
        for j in range(masked_score.shape[1]):
            if masked_score[i,j]==1:
                instance,instance_points=find_region(i,j)
                instance_list.append(instance)
                instance_points_list.append(instance_points)
                temp = np.sign(instance.copy()).astype(np.uint8)
                _,cnt,_ = cv2.findContours(temp, 1,2)
                if len(cnt) > 1:
                    print('more than one cnt')
                    print(cnt)
                    for cnt_ in cnt:
                        convex_hulls.append(cnt_)
                else:
                    convex_hulls.append(cnt[0])
    return instance_list,instance_points_list,convex_hulls

def totaltext_index():
    import json
    with open('/home/rjq/data_cleaned/data_cleaned/totaltext_algo3_128/test_index.json', 'r') as f:
        index = json.load(f)
    return index

def get_cnts(mat):
    cnts = []
    for i in range(len(mat['polygt'])):
        temp = []
        for x, y in zip(mat['polygt'][i][1][0], mat['polygt'][i][3][0]):
            temp.append([x,y])
        temp = np.expand_dims(np.array(temp), 1).astype(np.int32)
        cnts.append(temp)
    cnts_ = []
    for cnt in cnts:
        if len(cnt) >= 3:
            cnts_.append(cnt)
    return cnts_

def save_heatmap(save_name, map):
    if np.max(map) != 0.0 or np.max(map) != 0:
        cv2.imwrite(save_name, map.astype(np.uint8)*255/np.max(map))
    else:
        cv2.imwrite(save_name, map.astype(np.uint8))

def heatmap(output):
    score=output[:,:,0] # 0/1 mask
    theta=output[:,:,1]
    curvature=output[:,:,2]
    radius=output[:,:,3]
    mask=output[:, :, 4] # 0/1 mask
    masked_score=score*mask # multiplication is faster and logical '&' here.
    return score, theta, curvature, radius, mask, masked_score

def compress(output):
    score=(output[:,:,0]<output[:,:,1]).astype(output.dtype) # 0/1 mask
    theta=output[:,:,2]
    curvature=output[:,:,3]
    radius=output[:,:,4]
    mask=(output[:,:,5]<output[:,:,6]).astype(output.dtype)
    masked_score=score*mask
    output = np.stack((score, theta, curvature, radius, mask, masked_score), -1)
    return output

def main(args):

    mapnames = []
    index = totaltext_index()

    for name in os.listdir(INPUT_DIR+args.input_dir):
        if '_maps.npy' in name:
            mapnames.append(name)
    if args.check_num > len(mapnames) or args.check_num == 0:
        args.check_num = len(mapnames)
    for mapname in mapnames[:args.check_num]:
        imname = index[str(int(mapname.strip('_maps.npy')))]
        im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.jpg')
        mat  = sio.loadmat(TOTALTEXT_DIR+'groundtruth_text/Groundtruth/Polygon/Test/poly_gt_'+imname+'.mat')
        cnts = get_cnts(mat)
        im, cnts = resize(im, cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
        cv2.imwrite(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'.jpg', im)

        output = np.load(INPUT_DIR+args.input_dir+mapname)
        if output.shape[2] > 5:
            output = compress(output)
        score, theta, curvature, radius, mask, masked_score = heatmap(output)
        save_heatmap(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_score.jpg', score)
        save_heatmap(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_theta.jpg', theta)
        save_heatmap(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_curvature.jpg', curvature)
        save_heatmap(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_radius.jpg', radius)
        save_heatmap(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_mask.jpg', mask)
        save_heatmap(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_masked_score.jpg', masked_score)
        instance_list, instance_points_list, convex_hulls = reconstruction(output)
        instance_list_with_hulls = []
        for i in range(len(instance_list)):
            instance = np.tile(np.expand_dims(instance_list[i],2), [1,1,3])*255
            instance = cv2.drawContours(instance, [convex_hulls[i]], -1, (0,0,255), 1)
            instance_list_with_hulls.append(instance)
        print(len(instance_list_with_hulls))
        #cv2.imwrite(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_summary'+'.jpg', np.sum(instance_list_with_hulls, 0))
        hulls = np.zeros(list(output.shape[:2])+[3])
        hulls = cv2.drawContours(hulls, convex_hulls, -1, (0,0,255), 1)
        hulls = cv2.drawContours(hulls, cnts, -1, (255,255,255), 1)
        cv2.imwrite(SAVE_DIR+args.output_dir+mapname.strip('_maps.npy')+'_box'+'.jpg', hulls)

if __name__ == '__main__':
    args = get_args()
    args = check(args)
    print('only support the evaluation for totaltext_algo3_128 now')
    main(args)

