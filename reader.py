import scipy.io as sio
import numpy as np
import cv2
SYNTHTEXT_DIR = '/home/rjq/data/SynthText/SynthText/'
def data_loader(data_set,start_point,end_point):
    """
    as defined in data_cruncher._loader_initialization
    however this function should only works as a wraper, taking outputs from the following sub-functions
    :param data_set:
    :param start_point:
    :param end_point:
    :return: a generaror
    """
    dataset={'SynthText':SynthText_loader} #and  etc.

    return dataset[data_set](start_point,end_point)

##generators:

def SynthText_loader(start_point, end_point):
    """
    generator
    :param start_point: int
    :param end_point: int
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    gt = sio.loadmat(SYNTHTEXT_DIR+'gt.mat')
    print('total len of synthtext: '+len(gt['imnames'][0]))

    for index in range(start_point, end_point):
        imname = gt['imnames'][0][index][0]
        origin = cv2.imread('/home/rjq/data/SynthText/SynthText/'+imname)
        origin = np.array(origin, np.uint8)
        assert origin.shape[2] = 3
        word_cnts = np.transpose(gt['wordBB'][0][index], (2,1,0))
        char_cnts = np.transpose(gt['charBB'][0][index], (2,1,0))
        char_cnts = [np.array(char_cnt, np.float32) for char_cnt in char_cnts]
        word_cnts = [np.array(word_cnt, np.float32) for word_cnt in word_cnts]
        cnts = [char_cnts, word_cnts]
        yield {'img_name': imname,
               'img': origin,
               'contour': cnts}



def Totaltext_loader(start_point,end_point):
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