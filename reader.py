import scipy.io as sio
import numpy as np
import cv2
import os
import math
import csv

SYNTHTEXT_DIR = '/home/rjq/data/SynthText/SynthText/'
TOTALTEXT_DIR = '/home/rjq/data/Total-Text-Dataset/Download/'
MSRA_DIR ='/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/'


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

def SynthText_loader(patch_num, n_th_patch):
    '''
    :param patch_num:
    :param n_th_patch:
    :param is_train:
    :return:
    '''
    gt = sio.loadmat(SYNTHTEXT_DIR+'gt.mat')
    pic_num = len(gt['imnames'][0])
    print(pic_num)
    patch_length = pic_num//patch_num+1
    start_point = n_th_patch*patch_length
    if (n_th_patch+1)*patch_length > pic_num:
        end_point = pic_num
    else:
        end_point = (n_th_patch+1)*patch_length
    print(start_point,end_point)
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
        txt = gt['txt'][0][index].tolist()
        txt = [text.strip() for text in txt]

        chars = []
        for line in txt:
            for sub_line in line.split():
                temp = []
                for char in list(sub_line):
                    if char not in ('\n',):
                        temp.append(char)
                chars.append(temp)

        yield {'img_index': index,
               'img_name': imname,
               'img': origin,
               'contour': cnts,
               'chars': chars}

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
        patch_length = pic_num // patch_num+1
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
            origin = np.array(origin, np.uint8)
            cnts = [np.array(cnt, np.float32) for cnt in cnts]
            yield {'img_index': index,
                   'img_name': imname,
                   'img': origin,
                   'contour': cnts}

    else:
        imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR + 'totaltext/Images/Test')]
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num//patch_num+1
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
            origin, cnts = validate(origin, cnts)
            origin = np.array(origin, np.uint8)
            cnts = [np.array(cnt, np.float32) for cnt in cnts]
            yield {'img_index': index,
                   'img_name': imname,
                   'img': origin,
                   'contour': cnts}

def MSRA_TD_500_loader(patch_num, n_th_patch, is_train):
    '''
    :param patch_num:
    :param n_th_patch:
    :param is_train:
    :return:
    '''
    def get_cnts_msra(textes):
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

    if is_train:
        imnames = list(set([name.split('.')[0] for name in os.listdir(MSRA_DIR + 'train/')]))
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num//patch_num+1
        start_point = n_th_patch*patch_length
        if (n_th_patch+1)*patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch+1)*patch_length

        for index in range(start_point, end_point):
            print(index)
            imname = imnames[index]
            origin = cv2.imread(MSRA_DIR+'train/'+imname+'.JPG')
            if origin is None:
                print(imname + ' is missed')
                continue
            textes = [text.split() for text in open(MSRA_DIR+'train/'+imname+'.gt', 'r').readlines()]
            if len(textes) == 0:
                print('cnt for '+imname+'is missed')
                continue
            cnts = get_cnts_msra(textes)
            origin, cnts = validate(origin, cnts)
            yield {'img_index': index,
                   'img': origin,
                   'contour': cnts}

    else:
        imnames = list(set([name.split('.')[0] for name in os.listdir(MSRA_DIR + 'test/')]))
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num // patch_num + 1
        start_point = n_th_patch * patch_length
        if (n_th_patch + 1) * patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch + 1) * patch_length

        for index in range(start_point, end_point):
            imname = imnames[index]
            origin = cv2.imread(MSRA_DIR + 'test/' + imname + '.JPG')
            if origin is None:
                print(imname + ' is missed')
                continue
            textes = [text.split() for text in open(MSRA_DIR + 'test/' + imname + '.gt', 'r').readlines()]
            if len(textes) == 0:
                print('cnt for ' + imname + 'is missed')
                continue
            cnts = get_cnts_msra(textes)
            origin, cnts = validate(origin, cnts)
            yield {'img_index': index,
                   'img': origin,
                   'contour': cnts}

class _icdar_loader:
    _imgdata_icdar2017 = [
        (
            0, 7200,
            ["/home/zwj/ICDAR2017_MLT/training/img_{}.jpg", "/home/zwj/ICDAR2017_MLT/training/img_{}.png"],
            "/home/zwj/ICDAR2017_MLT/training_gt/gt_img_{}.txt",
            list(range(1, 7201))
        ),
        (
            7200, 9000,
            ["/home/zwj/ICDAR2017_MLT/validation/img_{}.jpg", "/home/zwj/ICDAR2017_MLT/validation/img_{}.png"],
            "/home/zwj/ICDAR2017_MLT/validation_gt/gt_img_{}.txt",
            list(range(1, 1801))
        ),
    ]

    _imgdata_icdar2017_rctw = [
        (
            0, 8034,
            ["/home/zwj/ICDAR2017_RCTW/train/image_{}.jpg", "/home/zwj/ICDAR2017_RCTW/train/image_{}.png"],
            "/home/zwj/ICDAR2017_RCTW/train/image_{}.txt",
            list(range(8034))
        ),
    ]

    _imgdata_icdar2015 = [
        (
            0, 1000,
            ["/home/rjq/data/ICDAR2015/ch4_training_images/img_{}.jpg", "/home/rjq/data/ICDAR2015/ch4_training_images/img_{}.png"],
            "/home/rjq/data/ICDAR2015/ch4_training_localization_transcription_gt/gt_img_{}.txt",
            list(range(1, 1001))
        ),
    ]

    _imgdata_icdar2013 = [
        (
            0, 229,
            ["/home/rjq/data/ICDAR2013/Challenge2_Training_Task12_Images/{}.jpg", "/home/rjq/data/ICDAR2013/Challenge2_Training_Task12_Images/{}.png"],
            "/home/rjq/data/ICDAR2013/Challenge2_Training_Task1_GT/gt_{}.txt",
            list(range(100, 329))
        ),
        (
            229, 462,
            ["/home/rjq/data/ICDAR2013/Challenge2_Test_Task12_Images/img_{}.jpg", "/home/rjq/data/ICDAR2013/Challenge2_Test_Task12_Images/img_{}.png"],
            "/home/rjq/data/ICDAR2013/Challenge2_Test_Task1_GT/gt_img_{}.txt",
            list(range(1, 234))
        ),
    ]

    _imgdata_td500 = [
        (
            0, 300,
            ["/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/train/IMG_{:04}.JPG"],
            "/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/train/IMG_{:04}.gt",
            [2165,1685,570,2213,809,821,2011,613,571,2199,2172,759,1862,1692,1645,1916,1719,1725,611,605,1724,1730,2205,1678,1687,758,764,748,2163,1683,827,628,601,826,1709,577,1641,1872,1866,2160,2174,1904,1723,818,2028,2014,603,617,602,2029,1736,1905,1939,2112,63,1619,1625,1786,1989,842,665,1547,1553,658,664,1591,472,1778,1977,506,1817,738,704,2113,1815,504,1975,1949,1785,1593,855,1550,896,1544,1545,1579,840,1586,1592,868,1753,1960,511,707,918,2100,515,850,1596,1582,893,2060,1569,2075,1540,2049,892,845,1756,1965,1971,514,1805,2101,702,64,728,2103,700,1783,489,1797,1967,1754,1768,476,2088,1595,884,890,660,2077,2062,649,885,463,1966,81,917,903,2127,730,1823,1957,1758,452,1770,687,650,1572,1567,2046,692,2085,2091,686,1771,1995,1822,719,916,2130,2124,733,1808,531,486,451,1983,479,848,690,2093,653,2078,1571,1570,1558,2051,652,861,849,2086,1799,1941,487,1955,518,530,1809,1835,726,722,1616,1951,497,1986,865,859,656,2055,2040,1549,1561,2097,858,694,2083,864,870,455,1763,469,496,1824,723,910,709,735,1832,1826,1615,1601,1629,523,1946,1761,1991,457,2081,669,899,655,1576,1562,697,2080,873,456,1748,1760,495,481,1947,1600,1614,907,784,753,747,1883,1673,1667,1920,814,1539,155,2024,633,2031,626,815,183,1712,1935,1672,746,752,791,787,2184,750,1705,2224,1922,792,769,594,1701,2208,812,635,621,620,608,1714,595,795,30,597,1676,810,2222,541,1677,582,596,1688,780,794]
        ),
        (
            300, 500,
            ["/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/test/IMG_{:04}.JPG"],
            "/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/test/IMG_{:04}.gt",
            [799,1691,1652,1646,1726,1732,612,607,2010,820,765,2166,1679,836,638,604,2013,610,1718,1903,599,770,760,2177,1867,1654,1668,2215,833,172,2002,1721,1696,763,1864,1657,830,616,158,831,1722,1865,711,2106,739,1802,513,507,1751,2099,103,671,659,1546,670,1626,1791,1587,672,666,1578,667,698,1627,505,1814,1800,59,1970,1964,1757,449,475,461,844,887,2074,1568,2061,886,851,1811,1839,716,2115,714,1620,462,1581,1556,675,2076,1557,891,1543,846,477,1972,1621,1806,2102,485,1943,491,1994,1764,1599,2090,888,2047,678,1598,1605,1836,2126,80,1607,1954,1940,445,1767,2044,1564,875,478,1766,1772,1969,2125,1825,520,1789,468,1992,2257,2082,2096,2069,2041,680,482,521,509,2120,912,721,1952,1749,866,2095,1563,898,668,1953,1628,790,592,545,2030,2018,2025,829,1706,1869,1699,793,1937,1923,803,156,2033,2032,625,802,2218,1936,1671,779,745,2181,1846,1675,1926,1715,1729,2220,807,2221,1933,1674,781,742,1689,554,839,2009,2035,2034,2008,2183]
        ),
    ]

    @staticmethod
    def _processcontour_201517(csvfile):
        cnts = []
        for x1, y1, x2, y2, x3, y3, x4, y4, *_ in csv.reader(csvfile):
            x1, y1, x2, y2, x3, y3, x4, y4 = (float(x.strip('\ufeff')) for x in (x1, y1, x2, y2, x3, y3, x4, y4))
            cnts.append(np.array([ [[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]], dtype=np.float))
        return cnts

    @staticmethod
    def _processcontour_2013(csvfile):
        cnts = []
        for left, top, right, bottom, *_ in csv.reader(csvfile, delimiter=' '):
            left, top, right, bottom = (float(x.strip(',')) for x in (left, top, right, bottom))
            cnts.append(np.array([ [[left, top]], [[left, bottom]], [[right, bottom]], [[right, top]]], dtype=np.float))
        return cnts

    @staticmethod
    def _processcontour_td500(csvfile):
        cnts = []
        for _, _, x, y, w, h, t in csv.reader(csvfile, delimiter=' '):
            x, y, w, h, t = (float(x.strip(',')) for x in (x, y, w, h, t))
            w /= 2
            h /= 2
            x += w
            y += h
            ct = math.cos(t)
            st = math.sin(t)
            wdx = w * ct
            wdy = w * st
            hdx = h * st
            hdy = -h * ct
            cnts.append(np.array([
                [[x + wdx + hdx, y + wdy + hdy]],
                [[x - wdx + hdx, y - wdy + hdy]],
                [[x - wdx - hdx, y - wdy - hdy]],
                [[x + wdx - hdx, y + wdy - hdy]]], dtype=np.float))
        return cnts

    @staticmethod
    def _load_icdar(imgdata, index, processor_contour):
        for beginindex, endindex, imgnames_t, gtname_t, files in imgdata:
            if index >= beginindex and index < endindex:
                i = files[index - beginindex]
                havefile = False
                for imgname_t in imgnames_t:
                    imgname = imgname_t.format(i)
                    print(imgname)
                    if os.path.exists(imgname):
                        havefile = True
                        break
                assert havefile
                img = cv2.imread(imgname)
                assert len(img.shape) == 3
                assert img.shape[2] == 3
                imname = "img_{}".format(index)
                gtname = gtname_t.format(i)
                with open(gtname) as gtfile:
                    cnts = processor_contour(gtfile)

                return {'img_index': index,
                'img_name': imname,
                'img': img,
                'contour': cnts}
        assert False

    @staticmethod
    def _loader(totallen, imgdata, processor_contour, patch_num, n_th_patch):
        patch_size = math.ceil(totallen / patch_num)
        for index in range(n_th_patch * patch_size, (n_th_patch+1) * patch_size):
            yield _icdar_loader._load_icdar(imgdata, index, processor_contour)

def ICDAR2017_loader(patch_num, n_th_patch):
    return _icdar_loader._loader(9000, _icdar_loader._imgdata_icdar2017, _icdar_loader._processcontour_201517, patch_num, n_th_patch)

def ICDAR2017RCTW_loader(patch_num, n_th_patch):
    return _icdar_loader._loader(8034, _icdar_loader._imgdata_icdar2017_rctw, _icdar_loader._processcontour_201517, patch_num, n_th_patch)

def ICDAR2015_loader(patch_num, n_th_patch):
    return _icdar_loader._loader(1000, _icdar_loader._imgdata_icdar2015, _icdar_loader._processcontour_201517, patch_num, n_th_patch)

def ICDAR2013_loader(patch_num, n_th_patch):
    return _icdar_loader._loader(462, _icdar_loader._imgdata_icdar2013, _icdar_loader._processcontour_2013, patch_num, n_th_patch)

def TD500_loader(patch_num, n_th_patch):
    return _icdar_loader._loader(500, _icdar_loader._imgdata_td500, _icdar_loader._processcontour_td500, patch_num, n_th_patch)

###################
# # Test for ICDAR2017_loader, ICDAR2017RCTW_loader, ICDAR2015_loader, ICDAR2013_loader, TD500_loader
# from itertools import chain
# import sys
# for img in chain(ICDAR2017_loader(1, 0), ICDAR2017RCTW_loader(1, 0), ICDAR2015_loader(1, 0), ICDAR2013_loader(1, 0), TD500_loader(1, 0)):
#     print(img['img_index'], img['img_name'], img['img'].shape, len(img['contour']))
# sys.exit(0)
###################

if __name__ == '__main__':
    import pickle
    from multiprocessing import Pool
    from multiprocessing import Process

    PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    generators = {'totaltext': Totaltext_loader}


    def othertext_to_pickle(save_dir, patch_num, n_th_patch, is_train, dataset):
        save_dir = save_dir.strip('/')
        save_dir = save_dir + '/'
        if not os.path.exists(PKL_DIR+save_dir):
           os.mkdir(PKL_DIR+save_dir)
        save_path = PKL_DIR+save_dir
        count = 0
        generator = generators[dataset]

        for res in generator(patch_num, n_th_patch, is_train):
            count += 1
            print('processing ' +str(count))
            img_index = res['img_index']
            img_name = res['img_name']
            img = res['img']
            contour = res['contour']
            img = np.array(img, np.uint8)
            contour = [np.array(cnt, np.float32) for cnt in contour]

            data_instance={
                'img_name':img_name,
                'img':img,
                'contour':contour,
                'is_text_cnts': True
            }

            pickle.dump(data_instance,open(os.path.join(save_path,'{}.bin'.format((img_index))),'wb'))


    def synthtext_to_pickle(save_dir, patch_num, n_th_patch):
        save_dir = save_dir.strip('/')
        save_dir = save_dir + '/'
        if not os.path.exists(PKL_DIR+save_dir):
            os.mkdir(PKL_DIR + save_dir)
        save_path = PKL_DIR + save_dir

        count = patch_num*n_th_patch
        for res in SynthText_loader(patch_num, n_th_patch):
            count += 1
            print('processing ' +str(count))
            img_index = res['img_index']
            img_name = res['img_name']
            img = res['img']
            contour = res['contour']
            chars = res['chars']
            char_contour, word_contour = contour
            img = np.array(img, np.uint8)
            char_contour = np.array(char_contour, np.float32)
            word_contour = np.array(word_contour, np.float32)
            contour = [char_contour, word_contour]

            data_instance = {
                'img_name': img_name,
                'img': img,
                'contour': contour,
                'is_text_cnts': False,
                'chars': chars
            }

            pickle.dump(data_instance, open(os.path.join(save_path, '{}.bin'.format((img_index))), 'wb'))


    patch_num = 20
    p=Pool(patch_num)
    # p.apply_async(othertext_to_pickle, args=('totaltext_train/', 1, 0, True, 'totaltext'))
    # p.apply_async(othertext_to_pickle, args=('totaltext_test/', 1, 0, False, 'totaltext'))
    for i in range(patch_num):
        p.apply_async(synthtext_to_pickle,args=('synthtext_chars/', patch_num, i))
    p.close()
    p.join()

    # jobs = []
    # patch_num = 20
    # for i in range(patch_num):
    #     jobs.append(Process(target=synthtext_to_pickle, args=('synthtext_chars/', patch_num, i)))
    # for job in jobs:
    #     job.start()
    # for job in jobs:
    #     job.join()




    ############ codes below are for tfrecord ##############
    # def _bytes_feature(value):
    #     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    # def _int64_feature(value):
    #     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    #
    # def _pad_cnt(cnt, cnt_point_max):
    #     new = []
    #     for cnt_ in cnt:
    #         if len(cnt_) < cnt_point_max:
    #             new.append(np.concatenate((cnt_, np.zeros([cnt_point_max-len(cnt_), 1, 2])), 0))
    #         else:
    #             new.append(cnt_)
    #     return new
    #
    # def othertext(save_dir, patch_num, n_th_patch, is_train, dataset):
    #     print('start')
    #     save_dir = save_dir.strip('/')
    #     save_dir = save_dir + '/'
    #     if not os.path.exists(TFRECORD_DIR+save_dir):
    #         os.mkdir(TFRECORD_DIR+save_dir)
    #     if is_train:
    #         tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_'+dataset+'_train.tfrecords'
    #     else:
    #         tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_'+dataset+'_test.tfrecords'
    #
    #     writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    #     print('get writer')
    #     count = 0
    #     generators = {'totaltext': Totaltext_loader}
    #                   # 'msra': MSRA_TD_500_loader}
    #     generator = generators[dataset]
    #     print(generator)
    #
    #     for res in generator(patch_num, n_th_patch, is_train):
    #         count += 1
    #         print('processing ' +str(count))
    #         img_index = res['img_index']
    #         img = res['img']
    #         img = np.array(img, np.uint8)
    #         img_row = img.shape[0]
    #         img_col = img.shape[1]
    #         contour = res['contour']
    #         cnt_point_num = np.array([len(contour[i]) for i in range(len(contour))], np.int64)
    #         cnt_num = len(contour)
    #         cnt_point_max = int(max(cnt_point_num))
    #         contour = _pad_cnt(contour, cnt_point_max)
    #         contour = np.array(contour, np.float32)
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'img_index': _int64_feature(img_index),
    #             'img': _bytes_feature(img.tostring()),
    #             'contour': _bytes_feature(contour.tostring()),
    #             'im_row': _int64_feature(img_row),
    #             'im_col': _int64_feature(img_col),
    #             'cnt_num': _int64_feature(cnt_num),
    #             'cnt_point_num': _bytes_feature(cnt_point_num.tostring()),
    #             'cnt_point_max': _int64_feature(cnt_point_max)
    #         }))
    #         writer.write(example.SerializeToString())
    #     writer.close()
    #
    # def synthtext(save_dir, patch_num, n_th_patch):
    #     save_dir = save_dir.strip('/')
    #     save_dir = save_dir + '/'
    #     if not os.path.exists(TFRECORD_DIR+save_dir):
    #         os.mkdir(TFRECORD_DIR+save_dir)
    #
    #     tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_synthtext.tfrecords'
    #     writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    #     count = 0
    #     for res in SynthText_loader(patch_num, n_th_patch, True):
    #         count += 1
    #         print('processing ' +str(count))
    #         img_index = res['img_index']
    #         img = res['img']
    #         img = np.array(img, np.uint8)
    #         img_row = img.shape[0]
    #         img_col = img.shape[1]
    #         contour = res['contour']
    #         char_contour, word_contour = contour
    #
    #         char_cnt_point_num = np.array([len(char_contour[i]) for i in range(len(char_contour))], np.int64)
    #         char_cnt_num = len(char_contour)
    #         char_cnt_point_max = int(max(char_cnt_point_num))
    #         char_contour = _pad_cnt(char_contour, char_cnt_point_max)
    #         char_contour = np.array(char_contour, np.float32)
    #
    #         word_cnt_point_num = np.array([len(word_contour[i]) for i in range(len(word_contour))], np.int64)
    #         word_cnt_num = len(word_contour)
    #         word_cnt_point_max = int(max(word_cnt_point_num))
    #         word_contour = _pad_cnt(word_contour, word_cnt_point_max)
    #         word_contour = np.array(word_contour, np.float32)
    #
    #
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'img_index': _int64_feature(img_index),
    #             'img': _bytes_feature(img.tostring()),
    #             'char_contour': _bytes_feature(char_contour.tostring()),
    #             'word_contour': _bytes_feature(word_contour.tostring()),
    #             'im_row': _int64_feature(img_row),
    #             'im_col': _int64_feature(img_col),
    #             'char_cnt_num': _int64_feature(char_cnt_num),
    #             'char_cnt_point_num': _bytes_feature(char_cnt_point_num.tostring()),
    #             'char_cnt_point_max': _int64_feature(char_cnt_point_max),
    #             'word_cnt_num': _int64_feature(word_cnt_num),
    #             'word_cnt_point_num': _bytes_feature(word_cnt_point_num.tostring()),
    #             'word_cnt_point_max': _int64_feature(word_cnt_point_max)
    #
    #         }))
    #         writer.write(example.SerializeToString())
    #     writer.close()
    #
    # def othertext_decoder(tfrecords_filename):
    #     record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    #     for string_record in record_iterator:
    #         example = tf.train.Example()
    #         example.ParseFromString(string_record)
    #
    #         img_index = int(example.features.feature['img_index']
    #                      .int64_list
    #                      .value[0])
    #         img_string = (example.features.feature['img']
    #                         .bytes_list
    #                         .value[0])
    #         contour_string = (example.features.feature['contour']
    #                         .bytes_list
    #                         .value[0])
    #         img_row = int(example.features.feature['im_row']
    #                      .int64_list
    #                      .value[0])
    #         img_col = int(example.features.feature['im_col']
    #                      .int64_list
    #                      .value[0])
    #         cnt_num = int(example.features.feature['cnt_num']
    #                      .int64_list
    #                      .value[0])
    #         cnt_point_num_string = (example.features.feature['cnt_point_num']
    #                         .bytes_list
    #                         .value[0])
    #         cnt_point_max = int(example.features.feature['cnt_point_max']
    #                      .int64_list
    #                      .value[0])
    #
    #         img_1d = np.fromstring(img_string, dtype=np.uint8)
    #         reconstructed_img = img_1d.reshape((img_row, img_col, -1))
    #         img = reconstructed_img
    #         cnt_point_num = np.fromstring(cnt_point_num_string, dtype=np.int64)
    #
    #         contour_1d = np.fromstring(contour_string, dtype=np.float32)
    #         reconstructed_contour = contour_1d.reshape((cnt_num, cnt_point_max, 1, 2))
    #         contour = []
    #         for i in range(cnt_num):
    #             contour.append(reconstructed_contour[i, :cnt_point_num[i], :, :])
    #         yield {'img_index': img_index,
    #                'img': img,
    #                'contour': contour}
    #
    # def synthtext_decoder(tfrecords_filename):
    #     record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    #     for string_record in record_iterator:
    #         example = tf.train.Example()
    #         example.ParseFromString(string_record)
    #
    #         img_index = int(example.features.feature['img_index']
    #                         .int64_list
    #                         .value[0])
    #         img_string = (example.features.feature['img']
    #             .bytes_list
    #             .value[0])
    #         char_contour_string = (example.features.feature['char_contour']
    #             .bytes_list
    #             .value[0])
    #         word_contour_string = (example.features.feature['word_contour']
    #             .bytes_list
    #             .value[0])
    #         img_row = int(example.features.feature['im_row']
    #                       .int64_list
    #                       .value[0])
    #         img_col = int(example.features.feature['im_col']
    #                       .int64_list
    #                       .value[0])
    #         char_cnt_num = int(example.features.feature['char_cnt_num']
    #                       .int64_list
    #                       .value[0])
    #         char_cnt_point_num_string = (example.features.feature['char_cnt_point_num']
    #             .bytes_list
    #             .value[0])
    #         char_cnt_point_max = int(example.features.feature['char_cnt_point_max']
    #                             .int64_list
    #                             .value[0])
    #         word_cnt_num = int(example.features.feature['word_cnt_num']
    #                       .int64_list
    #                       .value[0])
    #         word_cnt_point_num_string = (example.features.feature['word_cnt_point_num']
    #             .bytes_list
    #             .value[0])
    #         word_cnt_point_max = int(example.features.feature['word_cnt_point_max']
    #                             .int64_list
    #                             .value[0])
    #
    #         img_1d = np.fromstring(img_string, dtype=np.uint8)
    #         reconstructed_img = img_1d.reshape((img_row, img_col, -1))
    #         img = reconstructed_img
    #
    #         char_cnt_point_num = np.fromstring(char_cnt_point_num_string, dtype=np.int64)
    #         char_contour_1d = np.fromstring(char_contour_string, dtype=np.float32)
    #         char_reconstructed_contour = char_contour_1d.reshape((char_cnt_num, char_cnt_point_max, 1, 2))
    #         char_contour = []
    #         for i in range(char_cnt_num):
    #             char_contour.append(char_reconstructed_contour[i, :char_cnt_point_num[i], :, :])
    #
    #         word_cnt_point_num = np.fromstring(word_cnt_point_num_string, dtype=np.int64)
    #         word_contour_1d = np.fromstring(word_contour_string, dtype=np.float32)
    #         word_reconstructed_contour = word_contour_1d.reshape((word_cnt_num, word_cnt_point_max, 1, 2))
    #         word_contour = []
    #         for i in range(word_cnt_num):
    #             word_contour.append(word_reconstructed_contour[i, :word_cnt_point_num[i], :, :])
    #
    #         yield {'img_index': img_index,
    #                'img': img,
    #                'contour': [char_contour, word_contour]}