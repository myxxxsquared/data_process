import scipy.io as sio
import numpy as np
import cv2
import os
import math


SYNTHTEXT_DIR = '/home/rjq/data/SynthText/SynthText/'
TOTALTEXT_DIR = '/home/rjq/data/Total-Text-Dataset/Download/'


def data_loader(data_set,patch_num,n_th_patch, is_train):
    '''
    :param data_set:
    :param patch_num:
    :param n_th_patch:
    :param is_train: bool
    :return:
    '''
    dataset={'SynthText':SynthText_loader} #and  etc.

    return dataset[data_set](patch_num,n_th_patch, is_train)


##generators:
def validate(im, cnts):
    cols, rows = [], []
    for cnt in cnts:
        for i in range(len(cnt)):
            cols.append(cnt[i][0][0])
            rows.append(cnt[i][0][1])
    col_max = math.ceil(max(cols))
    row_max = math.ceil(max(rows))
    im_row, im_col = im.shape[0]-1, im.shape[1]-1
    if im_row < row_max:
        temp = np.zeros([row_max-im_row, im.shape[1], im.shape[2]])
        im = np.concatenate((im, temp), 0)
    if im_col < col_max:
        temp = np.zeros([im.shape[0], col_max-im_col, im.shape[2]])
        im = np.concatenate((im, temp), 1)
    return im, cnts


def SynthText_loader(patch_num, n_th_patch, is_train):
    '''
    :param patch_num:
    :param n_th_patch:
    :param is_train:
    :return:
    '''
    gt = sio.loadmat(SYNTHTEXT_DIR+'gt.mat')
    pic_num = len(gt['imnames'][0])
    patch_length = pic_num//patch_num
    start_point = n_th_patch*patch_length
    if (n_th_patch+1)*patch_length > pic_num:
        end_point = pic_num
    else:
        end_point = (n_th_patch+1)*patch_length

    for index in range(start_point, end_point):
        imname = gt['imnames'][0][index][0]
        origin = cv2.imread('/home/rjq/data/SynthText/SynthText/'+imname)
        origin = np.array(origin, np.uint8)
        assert origin.shape[2] == 3

        word_cnts = gt['wordBB'][0][index]
        char_cnts = gt['charBB'][0][index]
        if len(word_cnts.shape) == 2:
            word_cnts = np.expand_dims(word_cnts, 2)
        if len(char_cnts.shape) == 2:
            char_cnts = np.expand_dims(char_cnts, 2)
        word_cnts = np.transpose(word_cnts, (2,1,0))
        char_cnts = np.transpose(char_cnts, (2,1,0))

        char_cnts = [np.array(char_cnt, np.float32) for char_cnt in char_cnts]
        word_cnts = [np.array(word_cnt, np.float32) for word_cnt in word_cnts]
        cnts = [char_cnts, word_cnts]
        yield {'img_index': index,
               'img': origin,
               'contour': cnts}


def Totaltext_loader(patch_num, n_th_patch, is_train):
    '''
    :param patch_num:
    :param n_th_patch:
    :param is_train: bool
    :return:
    '''
    def get_total_cnts(mat):
        cnts = []
        for i in range(len(mat['polygt'])):
            temp = []
            for x, y in zip(mat['polygt'][i][1][0], mat['polygt'][i][3][0]):
                temp.append([x,y])
            temp = np.expand_dims(np.array(temp), 1).astype(np.float32)
            cnts.append(temp)
        cnts_ = []
        for cnt in cnts:
            if len(cnt) >= 3:
                cnts_.append(np.array(cnt, np.float32))
        return cnts_

    if is_train:
        imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR + 'totaltext/Images/Train')]
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num // patch_num
        start_point = n_th_patch * patch_length
        if (n_th_patch + 1) * patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch + 1) * patch_length

        for index in range(start_point, end_point):
            imname = imnames[index]
            origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.jpg')
            if origin is None:
                origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.JPG')
            if origin is None:
                print(imname+ ' is missed')
                continue
            mat = sio.loadmat(TOTALTEXT_DIR + 'groundtruth_text/Groundtruth/Polygon/Train/poly_gt_' + imname + '.mat')
            cnts = get_total_cnts(mat)
            origin, cnts = validate(origin, cnts)
            yield {'img_index': index,
                   'img': origin,
                   'contour': cnts}

    else:
        imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR + 'totaltext/Images/Test')]
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num//patch_num
        start_point = n_th_patch*patch_length
        if (n_th_patch+1)*patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch+1)*patch_length

        for index in range(start_point, end_point):
            imname = imnames[index]
            origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.jpg')
            if origin is None:
                origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.JPG')
            if origin is None:
                print(imname + 'is missed')
                continue
            mat = sio.loadmat(TOTALTEXT_DIR + 'groundtruth_text/Groundtruth/Polygon/Test/poly_gt_' + imname + '.mat')
            cnts = get_total_cnts(mat)
            origin = validate(origin, cnts)
            yield {'img_index': index,
                   'img': origin,
                   'contour': cnts}



def MSRA_TD_500_loader(start_point,end_point):
    """
    generator
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass

def ICDAR2017_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass

def ICDAR2015_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass

def ICDAR2013_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass

def TD500_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass

if __name__ == '__main__':
    # for res in SynthText_loader(10, 2, False):
    #     print(res)
    TFRECORD_DIR = '/home/rjq/data_cleaned/tfrecord/'

    import tensorflow as tf
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    #synthtext
    tfrecords_filename = TFRECORD_DIR+'synthtext.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for res in Totaltext_loader(1, 0, True):
        img_index = res['img_index']
        img = res['img']
        img_row = img.shape[0]
        img_col = img.shape[1]
        contour = res['contour']
        cnt_point_num = np.array([len(contour[i]) for i in range(len(contour))])
        cnt_num = len(contour)
        contour = np.array(contour)

        example = tf.train.Example(feature=tf.train.Feature(feature={
            'im_name': _int64_feature(img_index),
            'img': _bytes_feature(img.tostring()),
            'contour': _bytes_feature(contour.tostring()),
            'im_row': _int64_feature(img_row),
            'im_col': _int64_feature(img_col),
            'cnt_num': _int64_feature(cnt_num),
            'cnt_point_num': _bytes_feature(cnt_point_num.tostring())
        }))

        writer.write(example.SerializeToString())
        print(cnt_point_num)
        print(cnt_num)
        print(contour.tostring())
        print('-'*10)
        break

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        im_name = (example.features.feature['img_index']
                                      .int64_list
                                      .value[0])
        img_string = (example.features.feature['img']
                                      .bytes_list
                                      .value[0])

        print('img_index', img_index)
        print('img_string', img)



    # for res in Totaltext_loader(1, 0, False):
    #     print(res)
    # for res in Totaltext_loader(1, 0, True):
    #     print(res)