from utils import validate, resize, get_maps
import scipy.io as sio
import numpy as np
import cv2
import argparse
import os
import random
import json
import multiprocessing as mp

RESIZE_PIC_ROW = 512
RESIZE_PIC_COL = 512
RESIZE_GT_ROW = 128
RESIZE_GT_COL = 128

SAVE_DIR = '/home/rjq/data_cleaned/data_cleaned/'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
if not os.path.exists(SAVE_DIR+'sample/'):
    os.mkdir(SAVE_DIR+'sample/')

SYNTEXT_DIR = '/home/rjq/data/SynthText/SynthText/'
TOTALTEXT_DIR = '/home/rjq/data/Total-Text-Dataset/Download/'
MSRA_DIR = '/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/'
TOTAL_ALGO = 3
SYN_ALGO = 3
MSRA_ALGO = 3

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Enter the name of datasets seperated with comma, candidates are totaltext, synthtext, icdar2013, icdar2015, msra')
    parser.add_argument('save_name', type=str, default='', help='save_name')
    parser.add_argument('--thread', type=int, default=20, help='Number of parallel threads')
    parser.add_argument('--check', type=int, default = True, help='Whether to check the output')
    parser.add_argument('--check_num', type=int, default = 20, help='The number of examples for checking')
    return parser.parse_args()

def check(args):
    if args.check == 0:
        args.check = False
    else:
        args.check = True
    if args.thread == 0:
        args.thread = 1
    if args.check:
        if args.check_num == 0:
            raise AttributeError('please enter check_num')
    save_name = args.save_name.strip('/')
    args.save_name = save_name+'/'
    if not os.path.exists(SAVE_DIR+args.save_name):
        os.mkdir(SAVE_DIR+args.save_name)
    if args.check:
        if not os.path.exists(SAVE_DIR+'sample/'+args.save_name):
            os.mkdir(SAVE_DIR+'sample/'+args.save_name)
    return args

def check_process(im, cnts, save_name, algo = 3):
    im, cnts = validate(im, cnts)
    im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'.jpg', im)
    np.save(SAVE_DIR+'sample/'+save_name+'.npy', im)
    im, cnts = resize(im,cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
    skel, maps = get_maps(im,cnts,algo)
    np.save(SAVE_DIR+'sample/'+save_name+'_maps.npy', maps)
    zero = np.zeros(im.shape[:2], dtype = np.uint8)
    box = cv2.drawContours(zero, cnts, -1, (255), 1)
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'_box.jpg', box)
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'_skel.jpg', skel)
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'_score.jpg',
                        maps[:,:,0]*255/(np.max(maps[:,:,0])+1))
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'_theta.jpg',
                        np.abs(maps[:,:,1])*255*2/np.pi)
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'_curvature.jpg',
                        maps[:,:,2]*255/(np.max(maps[:,:,2])+1))
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'_radius.jpg',
                        maps[:,:,3]*255/(np.max(maps[:,:,3])+1))
    cv2.imwrite(SAVE_DIR+'sample/'+save_name+'_mask.jpg',
                        maps[:,:,4]*255/(np.max(maps[:,:,4])+1))

def generate_process(im, cnts, save_name, algo = 3):
    im, cnts = validate(im, cnts)
    im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
    np.save(SAVE_DIR+save_name+'.npy', im)
    im, cnts = resize(im, cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
    skel, maps = get_maps(im, cnts, algo)
    np.save(SAVE_DIR+save_name+'_maps.npy', maps)


args = get_args()
args = check(args)

if args.data == 'synthtext':
    gt = sio.loadmat(SYNTEXT_DIR+'gt.mat')
    pic_num = len(gt['imnames'][0])
    index = {}

    for i in range(pic_num):
        index[i] = str(gt['imnames'][0][i][0])
    with open(SAVE_DIR+args.save_name+'index.json', 'w+') as f:
        json.dump(index, f)

    if args.check:
        check_set = set()
        assert args.check_num <= pic_num, 'check_num is too big'
        while len(check_set) < args.check_num:
            check_set.add(random.randint(0, pic_num - 1))
        for i in check_set:
            imname = str(gt['imnames'][0][i][0])
            cnts = gt['wordBB'][0][i].transpose().astype(np.int32)
            if len(cnts.shape) == 2: cnts = np.expand_dims(cnts, 0)
            cnts = list(np.expand_dims(cnts, 2))
            im = cv2.imread(SYNTEXT_DIR + imname)
            im, cnts = validate(im, cnts)
            im_save_name = '{:0>8d}'.format(i)
            check_process(im, cnts, args.save_name+im_save_name,SYN_ALGO)

    def job(i):
        imname = str(gt['imnames'][0][i][0])
        cnts = gt['wordBB'][0][i].transpose().astype(np.int32)
        if len(cnts.shape) == 2: cnts = np.expand_dims(cnts, 0)
        cnts = list(np.expand_dims(cnts, 2))
        im = cv2.imread(SYNTEXT_DIR + imname)
        im_save_name = '{:0>8d}'.format(i)
        generate_process(im, cnts, args.save_name+im_save_name,SYN_ALGO)

    pool = mp.Pool(args.thread)
    pool.map(job, range(pic_num))

elif args.data == 'totaltext':
    if not os.path.exists(SAVE_DIR+args.save_name+'Train/'):
        os.mkdir(SAVE_DIR+args.save_name+'Train/')
    if not os.path.exists(SAVE_DIR+args.save_name+'Test/'):
        os.mkdir(SAVE_DIR+args.save_name+'Test/')

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

    imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR+'totaltext/Images/Train')]
    pic_num = len(imnames)
    temp = {}
    for i in range(pic_num):
        temp[i] = imnames[i]
    with open(SAVE_DIR+args.save_name+'train_index.json', 'w+') as f:
        json.dump(temp, f)

    if args.check:
        check_set = set()
        assert args.check_num <= pic_num, 'check_num is too big'
        while len(check_set) < args.check_num:
            check_set.add(random.randint(0, pic_num-1))

        for i in check_set:
            imname = imnames[i]
            im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.jpg')
            if im is None: im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.JPG')
            if im is None:
                print(i, imname)
                raise AttributeError(imname+' is not found')
            mat  = sio.loadmat(TOTALTEXT_DIR+'groundtruth_text/Groundtruth/Polygon/Train/poly_gt_'+imname+'.mat')
            cnts = get_cnts(mat)
            im, cnts = validate(im, cnts)
            im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
            im_save_name = '{:0>8d}'.format(i)
            check_process(im, cnts, args.save_name + im_save_name, TOTAL_ALGO)

    #process train
    def train_job(i):
        imname = imnames[i]
        im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.jpg')
        if im is None: im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.JPG')
        if im is None:
            print(i, imname)
            raise AttributeError(imname + ' is not found')
        mat  = sio.loadmat(TOTALTEXT_DIR+'groundtruth_text/Groundtruth/Polygon/Train/poly_gt_'+imname+'.mat')
        cnts = get_cnts(mat)
        im, cnts = validate(im, cnts)
        im_save_name = '{:0>8d}'.format(i)
        generate_process(im, cnts, args.save_name + 'Train/'+ im_save_name, TOTAL_ALGO)


    #process test
    imnames_test = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR+'totaltext/Images/Test')]
    pic_num_test = len(imnames_test)

    temp = {}
    for i in range(pic_num_test):
        temp[i] = imnames_test[i]
    with open(SAVE_DIR+args.save_name+'test_index.json', 'w+') as f:
        json.dump(temp, f)

    def test_job(i):
        imname = imnames_test[i]
        im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.jpg')
        if im is None: im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.JPG')
        if im is None:
            print(i, imname)
            raise AttributeError(imname + ' is not found')
        mat  = sio.loadmat(TOTALTEXT_DIR+'groundtruth_text/Groundtruth/Polygon/Test/poly_gt_'+imname+'.mat')
        cnts = get_cnts(mat)
        im, cnts = validate(im, cnts)
        im_save_name = '{:0>8d}'.format(i)
        generate_process(im, cnts, args.save_name + 'Test/'+ im_save_name, TOTAL_ALGO)

    pool = mp.Pool(args.thread)
    pool.map(train_job, range(pic_num))
    pool.close()
    pool.join()

    pool = mp.Pool(args.thread)
    pool.map(test_job, range(pic_num_test))
    pool.close()
    pool.join()

elif args.data == 'msra':

    if not os.path.exists(SAVE_DIR+args.save_name+'Train/'):
        os.mkdir(SAVE_DIR+args.save_name+'Train/')
    if not os.path.exists(SAVE_DIR+args.save_name+'Test/'):
        os.mkdir(SAVE_DIR+args.save_name+'Test/')

    def get_cnts(textes):
        cnts = []
        def reverse_point(point):
            return (point[1], point[0])
        for text in textes:
            points = []
            text = [float(num) for num in text]
            x, y, w, h, theta = text[2], text[3], text[4], text[5], text[6]
            point1 = (x, y)
            point2 = (x+w, y)
            point3 = (x+w, y+h)
            point4 = (x, y+h)
            rotateMatrix = cv2.getRotationMatrix2D((x+w/2,y+h/2), -theta*180/np.pi,1)
            point1 = np.matmul(rotateMatrix, point1+(1,))
            point2 = np.matmul(rotateMatrix, point2+(1,))
            point3 = np.matmul(rotateMatrix, point3+(1,))
            point4 = np.matmul(rotateMatrix, point4+(1,))
            points.append([point1])
            points.append([point2])
            points.append([point3])
            points.append([point4])
            cnts.append(np.array(points).astype(np.int32))
        return cnts

    imnames = list(set([name.split('.')[0] for name in os.listdir(MSRA_DIR+'train/')]))
    pic_num = len(imnames)

    if args.check:
        check_set = set()
        assert args.check_num <= pic_num, 'check_num is too big'
        while len(check_set) < args.check_num:
            check_set.add(random.randint(0, pic_num-1))

        for i in check_set:
            imname = imnames[i]
            im = cv2.imread(MSRA_DIR+'train/'+imname+'.JPG')
            assert im is not None, str(imname)+' is None'
            textes = [text.split() for text in open(MSRA_DIR+'train/'+imname+'.gt', 'r').readlines()]
            if len(textes) == 0: continue
            cnts = get_cnts(textes)
            im_save_name = '{:0>8d}'.format(i)
            check_process(im, cnts, args.save_name+im_save_name, MSRA_ALGO)

    def train_job(i):
        imname = imnames[i]
        im = cv2.imread(MSRA_DIR+'train/'+imname+'.JPG')
        assert im is not None, str(imname)+' is None'
        textes = [text.split() for text in open(MSRA_DIR+'train/'+imname+'.gt', 'r').readlines()]
        if len(textes) == 0: return
        cnts = get_cnts(textes)
        im_save_name = '{:0>8d}'.format(i)
        generate_process(im, cnts, args.save_name + 'Train/'+ im_save_name, MSRA_ALGO)

    imnames_test = [name.split('.')[0] for name in os.listdir(MSRA_DIR+'test/')]
    pic_num_test = len(imnames)

    def test_job(i):
        imname = imnames_test[i]
        im = cv2.imread(MSRA_DIR+'test/'+imname+'.JPG')
        assert im is not None, str(imname)+' is None'
        textes = [text.split() for text in open(MSRA_DIR+'test/'+imname+'.gt', 'r').readlines()]
        if len(textes) == 0: return
        cnts = get_cnts(textes)
        im_save_name = '{:0>8d}'.format(i)
        generate_process(im, cnts, args.save_name + 'Test/'+ im_save_name, MSRA_ALGO)

    pool = mp.Pool(args.thread)
    pool.map(train_job, range(pic_num))
    pool.close()
    pool.join()

    pool = mp.Pool(args.thread)
    pool.map(test_job, range(pic_num_test))
    pool.close()
    pool.join()

else:
    raise NotImplementedError(args.data +' is not supported now')