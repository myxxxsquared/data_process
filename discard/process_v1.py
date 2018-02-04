from utils import validate, resize, get_maps
import scipy.io as sio
import numpy as np
import cv2
import argparse
import os
import threading
import math
import random
import multiprocessing

RESIZE_PIC_ROW = 512
RESIZE_PIC_COL = 512
RESIZE_GT_ROW = 128
RESIZE_GT_COL = 128

SAVE_DIR = '/home/rjq/data_cleaned/test/test_algo3/'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

SYNTEXT_DIR = '/home/rjq/data/SynthText/SynthText/'
TOTALTEXT_DIR = '/home/rjq/data/Total-Text-Dataset/Download/'
TOTAL_ALGO = 3
SYN_ALGO = 3
MSRA_DIR = '/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='Enter the name of datasets seperated with comma, candidates are totaltext, synthtext, icdar2013, icdar2015, msra')
    parser.add_argument('--save_dir', type=str, default='', help='SAVE_DIR')
    parser.add_argument('--thread', type=int, default=0, help='Number of parallel threads')
    parser.add_argument('--check', type=bool, default = False, help='Whether to check the output')
    parser.add_argument('--check_num', type=int, default = 0, help='The number of examples for checking')
    return parser.parse_args()

def check(args):
    if args.thread == 0:
        args.thread = 1
    if args.check:
        if args.check_num == 0:
            raise AttributeError('please enter check_num')
    if args.save_dir != '':
        SAVE_DIR = args.save_dir
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
    return args


def check_process(im, cnts, dataname, save_name, algo = 0):
    im, cnts = validate(im, cnts)
    im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'.jpg', im)
    np.save(SAVE_DIR+dataname+'/Train/'+save_name+'.npy', im)
    im, cnts = resize(im,cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
    skel, maps = get_maps(im,cnts,algo)
    np.save(SAVE_DIR+dataname+'/Train/'+save_name+'_maps.npy', maps)
    zero = np.zeros(im.shape[:2], dtype = np.uint8)
    box = cv2.drawContours(zero, cnts, -1, (255), 1)

    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'_box.jpg', box)
    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'_skel.jpg', skel)
    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'_score.jpg',
                        maps[:,:,0]*255/(np.max(maps[:,:,0])+1))
    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'_theta.jpg',
                        np.abs(maps[:,:,1])*255*2/np.pi)
    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'_curvature.jpg',
                        maps[:,:,2]*255/(np.max(maps[:,:,2])+1))
    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'_radius.jpg',
                        maps[:,:,3]*255/(np.max(maps[:,:,3])+1))
    cv2.imwrite(SAVE_DIR+'test/'+dataname+'/'+save_name+'_mask.jpg',
                        maps[:,:,4]*255/(np.max(maps[:,:,4])+1))

def generate_process(im, cnts, dataname, save_name, algo = 0):
    im, cnts = validate(im, cnts)
    im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
    np.save(SAVE_DIR + dataname+'/' + save_name + '.npy', im)
    im, cnts = resize(im, cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
    skel, maps = get_maps(im, cnts, algo)
    np.save(SAVE_DIR + dataname+'/' + save_name + '_maps.npy', maps)

def synthtext(args):
    if not os.path.exists(SAVE_DIR+'synthtext/'):
        os.mkdir(SAVE_DIR+'synthtext')
    if args.check:
        if not os.path.exists(SAVE_DIR+'test/'):
            os.mkdir(SAVE_DIR+'test')
        if not os.path.exists(SAVE_DIR+'test/synthtext/'):
            os.mkdir(SAVE_DIR+'test/synthtext')

    gt = sio.loadmat(SYNTEXT_DIR+'gt.mat')
    pic_num = len(gt['imnames'][0])
    
    def run_thread(start, end):
        for i in range(start, end):
            imname = str(gt['imnames'][0][i][0])
            cnts = gt['wordBB'][0][i].transpose().astype(np.int32)
            if len(cnts.shape) == 2: cnts = np.expand_dims(cnts, 0)
            cnts = list(np.expand_dims(cnts, 2))
            im = cv2.imread(SYNTEXT_DIR+imname)
            save_name = imname.split('/')[-1].split('.')[0]
            im, cnts = validate(im, cnts)
            im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
            np.save(SAVE_DIR+'synthtext/'+save_name+'.npy', im)
            im, cnts = resize(im,cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
            skel, maps = get_maps(im,cnts,SYN_ALGO)
            np.save(SAVE_DIR+'synthtext/'+save_name+'_maps.npy', maps)
    
    # check
    if args.check:
         check_set = set()
         assert args.check_num <= pic_num, 'check_num is too big'
         while len(check_set) < args.check_num:
             check_set.add(random.randint(0, pic_num-1))
         for i in check_set:
            imname = str(gt['imnames'][0][i][0])
            cnts = gt['wordBB'][0][i].transpose().astype(np.int32)
            if len(cnts.shape) == 2: cnts = np.expand_dims(cnts, 0)
            cnts = list(np.expand_dims(cnts, 2))
            im = cv2.imread(SYNTEXT_DIR+imname)
            save_name = imname.split('/')[-1].split('.')[0]
            im, cnts = validate(im, cnts)
            im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'.jpg', im)
            np.save(SAVE_DIR+'test/synthtext/'+save_name+'.npy', im)
            im, cnts = resize(im,cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
            skel, maps = get_maps(im,cnts,SYN_ALGO)
            np.save(SAVE_DIR+'test/synthtext/'+save_name+'_maps.npy', maps)
            zero = np.zeros(im.shape[:2], dtype = np.uint8)
            box = cv2.drawContours(zero, cnts, -1, (255), 1)

            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'_box.jpg', box)
            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'_skel.jpg', skel)
            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'_score.jpg',
                        maps[:,:,0]*255/(np.max(maps[:,:,0])+1))
            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'_theta.jpg',
                        np.abs(maps[:,:,1])*255*2/np.pi)
            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'_curvature.jpg',
                        maps[:,:,2]*255/(np.max(maps[:,:,2])+1))
            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'_radius.jpg',
                        maps[:,:,3]*255/(np.max(maps[:,:,3])+1))
            cv2.imwrite(SAVE_DIR+'test/synthtext/'+save_name+'_mask.jpg',
                        maps[:,:,4]*255/(np.max(maps[:,:,4])+1))
    # generate      
    interval = math.ceil(pic_num/args.thread)
    jobs = []
    for i in range(args.thread):
        start = i*interval
        if (i+1)*interval > pic_num: end = pic_num
        else: end = (i+1)*interval
        jobs.append(multiprocessing.Process(target=run_thread, args = (start,end)))
    for i in range(len(jobs)):
        jobs[i].start()
    for i in range(len(jobs)):
        jobs[i].join()
    print('finished processing synthtext')


def totaltext(args):
    if not os.path.exists(SAVE_DIR+'totaltext/'):
        os.mkdir(SAVE_DIR+'totaltext')
    if args.check:
        if not os.path.exists(SAVE_DIR+'test/'):
            os.mkdir(SAVE_DIR+'test')
        if not os.path.exists(SAVE_DIR+'test/totaltext/'):
            os.mkdir(SAVE_DIR+'test/totaltext')
    
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
    # process trian
    if not os.path.exists(SAVE_DIR+'totaltext/Train/'):
        os.mkdir(SAVE_DIR+'totaltext/Train')
    imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR+'totaltext/Images/Train')]
    pic_num = len(imnames)

    # check first
    if args.check:
        check_set = set()
        assert args.check_num <= pic_num, 'check_num is too big'
        while len(check_set) < args.check_num:
            check_set.add(random.randint(0, pic_num-1))
        #TODO
        check_set.add(imnames.index('img1045'))
        for i in check_set:
            imname = imnames[i]
            save_name = imname
            im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.jpg')
            if im is None: im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.JPG')
            if im is None:
                print(i, imname)
                continue
            mat  = sio.loadmat(TOTALTEXT_DIR+'groundtruth_text/Groundtruth/Polygon/Train/poly_gt_'+imname+'.mat')
            cnts = get_cnts(mat)
            im, cnts = validate(im, cnts)
            im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'.jpg', im)
            np.save(SAVE_DIR+'totaltext/Train/'+save_name+'.npy', im)
            im, cnts = resize(im,cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
            skel, maps = get_maps(im,cnts,TOTAL_ALGO)
            np.save(SAVE_DIR+'totaltext/Train/'+save_name+'_maps.npy', maps)
            zero = np.zeros(im.shape[:2], dtype = np.uint8)
            box = cv2.drawContours(zero, cnts, -1, (255), 1)

            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'_box.jpg', box)
            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'_skel.jpg', skel)
            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'_score.jpg',
                        maps[:,:,0]*255/(np.max(maps[:,:,0])+1))
            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'_theta.jpg',
                        np.abs(maps[:,:,1])*255*2/np.pi)
            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'_curvature.jpg',
                        maps[:,:,2]*255/(np.max(maps[:,:,2])+1))
            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'_radius.jpg',
                        maps[:,:,3]*255/(np.max(maps[:,:,3])+1))
            cv2.imwrite(SAVE_DIR+'test/totaltext/'+save_name+'_mask.jpg',
                        maps[:,:,4]*255/(np.max(maps[:,:,4])+1))
    # process train
    for i in range(pic_num):
        imname = imnames[i]
        save_name = imname
        im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.jpg')
        if im is None: im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.JPG')
        if im is None:
            print(i, imname)
            continue
        mat  = sio.loadmat(TOTALTEXT_DIR+'groundtruth_text/Groundtruth/Polygon/Train/poly_gt_'+imname+'.mat')
        cnts = get_cnts(mat)    
        im, cnts = validate(im, cnts)
        im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
        np.save(SAVE_DIR+'totaltext/Train/'+save_name+'.npy', im)
        im, cnts = resize(im,cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
        skel, maps = get_maps(im,cnts,TOTAL_ALGO)
        np.save(SAVE_DIR+'totaltext/Train/'+save_name+'_maps.npy', maps)
    # process test 
    if not os.path.exists(SAVE_DIR+'totaltext/Test/'):
        os.mkdir(SAVE_DIR+'totaltext/Test')
    imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR+'totaltext/Images/Test')]
    pic_num = len(imnames)

    for i in range(pic_num):
        imname = imnames[i]
        save_name = imname
        im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.jpg')
        if im is None: im = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.JPG')
        if im is None:
            print(i, imname)
            continue
        mat  = sio.loadmat(TOTALTEXT_DIR+'groundtruth_text/Groundtruth/Polygon/Test/poly_gt_'+imname+'.mat')
        cnts = get_cnts(mat)
        im, cnts = validate(im, cnts)
        im, cnts = resize(im, cnts, RESIZE_PIC_ROW, RESIZE_PIC_COL)
        np.save(SAVE_DIR+'totaltext/Test/'+save_name+'.npy', im)
        im, cnts = resize(im,cnts, RESIZE_GT_ROW, RESIZE_GT_COL)
        skel, maps = get_maps(im,cnts,TOTAL_ALGO)
        np.save(SAVE_DIR+'totaltext/Test/'+save_name+'_maps.npy', maps)

def msra(args):
    if not os.path.exists(SAVE_DIR+'msra/'):
        os.mkdir(SAVE_DIR+'msra')
    if args.check:
        if not os.path.exists(SAVE_DIR+'test/'):
            os.mkdir(SAVE_DIR+'test')
        if not os.path.exists(SAVE_DIR+'test/msra/'):
            os.mkdir(SAVE_DIR+'test/msra')
    if not os.path.exists(SAVE_DIR+'msra/Train/'):
        os.mkdir(SAVE_DIR+'msra/Train')

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
           
    # train first
    imnames = list(set([name.split('.')[0] for name in os.listdir(MSRA_DIR+'train/')]))
    pic_num = len(imnames)

    # check first
    if args.check:
        check_set = set()
        assert args.check_num <= pic_num, 'check_num is too big'
        while len(check_set) < args.check_num:
            check_set.add(random.randint(0, pic_num-1))
        for i in check_set:
            imname = imnames[i]
            save_name = imname
            im = cv2.imread(MSRA_DIR+'train/'+imname+'.JPG')
            assert im is not None, str(imname)+' is None'
            textes = [text.split() for text in open(MSRA_DIR+'train/'+imname+'.gt', 'r').readlines()]
            if len(textes) == 0: continue
            cnts = get_cnts(textes)
            check_process(im, cnts, 'msra', save_name, 0)
    
    # process train
    for i in range(pic_num):
        imname = imnames[i]
        save_name = imname
        im = cv2.imread(MSRA_DIR+'train/'+imname+'.JPG')
        assert im is not None, str(imname)+' is None'
        textes = [text.split() for text in open(MSRA_DIR+'train/'+imname+'.gt', 'r').readlines()]
        if len(textes) == 0: continue
        cnts = get_cnts(textes)
        generate_process(im, cnts, 'msra', 'Train/'+save_name, 0)

    # process test
    if not os.os.path.exists(SAVE_DIR+'msra/Test/'):
        os.mkdir(SAVE_DIR+'msra/Test')
    imnames = [name.split('.')[0] for name in os.listdir(MSRA_DIR+'test/')]
    pic_num = len(imnames)
    
    for i in range(pic_num):
        imname = imanes[i]
        save_name = imname
        im = cv2.imread(MSRA_DIR+'test/'+imname+'.JPG')
        assert im is not None, str(imname)+' is None'
        textes = [text.split() for text in open(MSRA_DIR+'test/'+imname+'.gt', 'r').readlines()]
        if len(textes) == 0: continue
        cnts = get_cnts(textes)
        generate_process(im, cnts, 'msra', 'Test/'+save_name, 0)

def main(args):
    datasets = [name.strip() for name in args.data.lower().split(',')]
    if 'synthtext' in datasets:
        synthtext(args)
    if 'totaltext' in datasets:
        totaltext(args)
    if 'msra' in datasets:
        msra(args)

if __name__ == '__main__':
    args = get_args()
    args = check(args)
    main(args)
