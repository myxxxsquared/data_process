#from .reader import data_loader  # <- for raw data loading
#from .util import *              # <- for data_augmentation
from .utils import get_maps
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('data_set', type=str, help='appoint a dataset to crunch')
parser.add_argument('start', type=int, help='starting point of the cruncher')
parser.add_argument('end', type=int, help='end point of the cruncher')
parser.add_argument('tf_record_path', type=str, help='appoint a path to store')
args = parser.parse_args()

class data_churn(object):
    def __init__(self, thickness=0.2, neighbor=5, crop_skel=1.0, *args,**kw):
        """
        initialize an instance
        :param kw: 'data_set': str, 'SynthText', 'totaltext', etc.
                     'start_point','end_point':int, indicating the starting point for the crunching process
               thickness: the thickness of the text center line
               neighbor: the range used for fit the theta
               crop_skel: the length for cropping the text center line (skeleton)
        """
        self.thickness = thickness
        self.neighbor = neighbor
        self.crop_skel =crop_skel
        pass

    def _loader_initialization(self):
        """
        initialize self with a data list and loader, the loader should be a generator and
        yields the following format:
        {'img_name':str,   original_name
        'img':np.uint8,
        'contour':List[the contour of each text instance],
        'type': 'char' or 'tl',
        'flag':if this is synthetext or not}
        for the contour of TI, it should be List[cnt] where x,y are float(cnt)
        for the contour of char_list, it should be List[List(cnt)] List(p1,p2,p3,p4)- list of char
        :return:
        """
        #self._data_loader=data_loader(self.data_set,self.start_point,self.end_point)
        pass

    def _data_augmentation(self,*args,**kw):
        """
        generator
        :param kw:

        {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}

        :return: yielding all augmented data one by one in order

        using function from .util(to be determined)

        """
        pass

    def _data_labeling(self, img_name, img, cnts, is_text_cnts, left_top, right_bottom, chars = None):
        '''
        :param img_name: pass to return directly, (to be determined, int or str)
        :param img: ndarray, np.uint8,
        :param cnts:
                if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2], order(col, row)
                if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
        :param is_text_cnts: bool
        :param left_top: for cropping
        :param right_bottom: for cropping
        :param chars:
                if is_text_cnts is True: None
                if is_text_cnts is False: a nested list storing the chars info for synthtext
        :return:
                img_name: passed down
                img: np.ndarray np.uint8
                maps: [TR, TCL, radius, cos_theta, sin_theta], all of them are 2-d array,
                TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
        '''

        skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
            get_maps(img, cnts, is_text_cnts, self.thickness, self.crop_skel, self.neighbor, chars)
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
        TR = TR[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
        TCL = TCL[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
        radius = radius[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
        cos_theta = cos_theta[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
        sin_theta = sin_theta[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
        img = img[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1],:]
        maps = [TR, TCL, radius, cos_theta, sin_theta]
        return img_name, img, maps


    def _data_generator_wrapper(self):
        """
        generator
        this function should work only as a wrapper and data label for the raw data generation.
        for synthtext: it returns what self.data_loader yields,
        for others: yields all DataAugmentation results one by one for each return generated by self.data_loader

        this function takes no params
        :return:
        """
        pass

    def data_to_tfrecord(self,tf_record_path):
        """
        this function uses all functions defined above to convert the data into tf_record located at tf_record_path
        :param tf_record_path:
        :return:
        """
        pass

if __name__=='__main__':
    #implement the main function for call data labeling conversion
    a=data_churn()## params
    a.data_to_tfrecord(args.tf_record_path)


