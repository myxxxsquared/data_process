#   to_be_determined
#   please refer to https://github.com/aleju/imgaug
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from random import shuffle,randint
import cv2
import copy


class DataAugmentor(object):
    """
    all data_augmentation defined below should take input as :
    {'img_name':str,   original_name
        'img':np.uint8,
        'contour':List[np.array(the contour of each text instance), (n,1,2)],
        'is_text_cnts': bool, true for cnts of boxes,
                            false for cnts of char}

    synthetext: no augment, char, False;
    others: augment, box

    while returning the same format.

    valuable data augmentation types:
        **crop, pad, flip,
        invert/add(overall/channel), add to hue and saturation, multiply,
        gaussian blur, average blur, median blur, bilateral blur,
        sharpen, emboss, edge detect,
        noise: guassian(overall/channel), dropout(pixel/channel, coarse,channel), salt&pepper
        norm: contrast
        gray scale
        **affine transformation#i should implement it myself, maybe

    """
    def __init__(self, *args, **kw):
        self.add_augmentation_list = [iaa.Add((-50, 50), per_channel=True),
                                 iaa.Add((-50, 50), per_channel=False),
                                 iaa.AddElementwise((-30, 30), per_channel=False),
                                 iaa.AddElementwise((-30, 30), per_channel=True),
                                 iaa.Invert(p=0.2, per_channel=True),
                                 iaa.Invert(p=0.2, per_channel=False),
                                 iaa.AddToHueAndSaturation((0, 80), True),
                                 iaa.Multiply((0.8, 1.2), per_channel=True),
                                 iaa.Multiply((0.8, 1.2), per_channel=False),
                                 iaa.MultiplyElementwise((0.8, 1.2), per_channel=True),
                                 iaa.MultiplyElementwise((0.8, 1.2), per_channel=False)
                                 ]

        self.blur_augmentation_list = [
            iaa.GaussianBlur((1, 3)),
            iaa.AverageBlur((1, 3)),
            iaa.MedianBlur((1, 3)),
            iaa.BilateralBlur((1, 3))
        ]

        self.noise_augmentation_list = [
            iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=True),
            iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=False),
            iaa.Dropout((0.05, 0.15), per_channel=False),
            iaa.Dropout((0.05, 0.15), per_channel=True),
            iaa.CoarseDropout((0.05, 0.15), size_percent=(0.5, 0.7)),
            iaa.SaltAndPepper((0.05, 0.15), per_channel=True),
            iaa.SaltAndPepper((0.05, 0.15), per_channel=False)
        ]

        self.other_augmentation_list = [
            iaa.Sharpen((0.9, 0.11), (0.8, 1.2)),
            iaa.Emboss((0.9, 0.11), (0.3, 1.6)),
            iaa.EdgeDetect((0, 0.4)),
            iaa.Grayscale((0, 1))
        ]


    def _get_seq(self,affine=False):
        if affine:
            affine = [
                iaa.Affine(rotate=360*np.random.rand()-180)   ,
                 iaa.Affine(shear=110*np.random.rand()-55),
                 iaa.Affine(scale={"x": 0.4*np.random.rand()+0.8, "y": 0.4*np.random.rand()+0.8})
            ]
            return iaa.Sequential(affine)

        shuffle(self.add_augmentation_list)
        add_augmentation_list = self.add_augmentation_list[:randint(0, 1)]
        shuffle(self.other_augmentation_list)
        other_augmentation_list = self.other_augmentation_list[:randint(0,1)]
        shuffle(self.blur_augmentation_list)
        blur_augmentation_list = self.blur_augmentation_list[:randint(0, 1)]
        shuffle(self.noise_augmentation_list)
        noise_augmentation_list = self.noise_augmentation_list[:randint(0, 1)]

        final_list=add_augmentation_list + noise_augmentation_list + blur_augmentation_list + other_augmentation_list

        if len(final_list)==0:
            final_list.append(iaa.Noop())

        return iaa.Sequential(final_list)

    @staticmethod
    def _key_points(image_shape, point_list):
        """
        feed cnt and return ia.KeypointsOnImage object
        :param point_list: np.array size=(n,1,4)
               image_shape
        :return:
        """
        keypoint_list = []
        for i in range(point_list.shape[0]):
            keypoint_list.append(ia.Keypoint(x=point_list[i, 0, 0], y=point_list[i, 0, 1]))
        return ia.KeypointsOnImage(keypoint_list,
                                   shape= ia.quokka(size=image_shape[:2]))

    @staticmethod
    def _enlarge(input_data):
        """
        resize image under 512P to 512P
        :param input_data:
        :return:
        return scaled img
        """
        rate=1
        if input_data['img'].shape[0] > input_data['img'].shape[1]:
            if True:#input_data['img'].shape[1] < 512:
                rate=512/input_data['img'].shape[1]
                seq = iaa.Sequential([
                    iaa.Scale({'height': "keep-aspect-ratio", 'width': 512}, 'cubic')
                ])
                input_data['img'] = seq.augment_image(input_data['img'])
        else:
            if True:#input_data['img'].shape[0] < 512:
                rate=512/input_data['img'].shape[0]
                seq = iaa.Sequential([
                    iaa.Scale({'height': 512, 'width': "keep-aspect-ratio"}, 'cubic')
                ])
                input_data['img'] = seq.augment_image(input_data['img'])

        input_data['contour']=[np.cast['int32'](contour*rate) for contour in input_data['contour']]
        return input_data

    def _pad(self,input_data):
        h = input_data['img'].shape[0]
        w = input_data['img'].shape[1]
        max_size = max([int(np.sqrt(np.power(h, 2) + np.power(w, 2))),
                        int(w+h*np.cos(11/36))
                        ]) + 5

        up = (max_size - h) // 2
        down = max_size - up - h
        left = (max_size - w) // 2
        right = max_size - left - w

        input_data['img'] = np.pad(input_data['img'], ((up,down), (left, right),(0,0)), mode='constant')
        print('before',input_data['contour'][0])
        print(up,left)
        input_data['contour'] = list(map(lambda x: np.stack([x[:, :, 0]+up, x[:, :, 1]+left], axis=-1),#x: np.array(n,1,2)
                                         input_data['contour']))
        print('after', input_data['contour'][0])
        return input_data

    def _pixel_augmentation(self, inputs):
        """
        1. pad a large black background
        2. doing augmentations that do not make affine transformation

        invert/add(overall/channel), add to hue and saturation, multiply,
        gaussian blur, average blur, median blur, bilateral blur,
        sharpen, emboss, edge detect,
        noise: guassian(overall/channel), dropout(pixel/channel, coarse,channel), salt&pepper
        norm: contrast
        gray scale

        return: padded + augmented image

        """
        input_data = copy.deepcopy(inputs)

        input_data['img'] = self._get_seq().augment_image(input_data['img'])
        cv2.imshow('show', input_data['img'])
        #x=input('enter to see next step')

        return input_data

    def _affine_transformation(self, inputs):
        """
        affine types include:
            1. scaling
            2. rotate
            3. shear
            4. aspect ratios
        doing affine transformation
        :param args:
        :param kw:
        :return:
        """
        input_data=copy.deepcopy(inputs)
        transformer=self._get_seq(affine=True)
        input_data['img'] = transformer.augment_image(input_data['img'])
        for p,cnt in enumerate(input_data['contour']):
            input_data['contour'][p]=transformer.augment_keypoints([self._key_points(image_shape=input_data['img'].shape,point_list=cnt)])[0]
            input_data['contour'][p]=[np.array([[int(keypoints.y),int(keypoints.x)]])for keypoints in input_data['contour'][p].keypoints]
            input_data['contour'][p]=np.stack(input_data['contour'][p],axis=0)
        return input_data

    def _crop_flip_pad(self, input_data):
        """
        flip, pad, crop, as final states.
        :param input_data:
        :return:
        """
        shape = input_data['img'].shape
        center = (shape[0]//2,shape[1]//2)
        return (center[0]+randint(-100,100),center[1]+randint(-100,100))

    def augment(self, input_data,augment_rate=100):
        """

        :param input_data:
               Dict{'img_name':str,   original_name
                'img':np.uint8,
                'contour':List[the contour of each text instance],
                'type': 'char' or 'tl',
                'is_text_cnts':if this is synthetext or not}
        :return:
        Dict{'img_name':str,   original_name
            'img':np.uint8,
            'contour':List[the contour of each text instance],
            'type': 'char' or 'tl',
            'flag':if this is synthetext or not,
            'left_top': tuple (x, y), x is row, y is col, please be careful about the order,
                 'right_bottom': tuple (x, y), x is row, y is col}
        """
        yield copy.deepcopy(input_data), (0, 0)
        input_data = self._enlarge(input_data)
        yield copy.deepcopy(input_data), (0, 0)
        if not input_data['is_text_cnts']:
            yield input_data,(0,0)
            return

        input_data = self._pad(input_data)
        yield copy.deepcopy(input_data), (0, 0)

        input_data['img']=np.transpose(input_data['img'],axes=[1,0,2])#？？？

        for i in range(augment_rate):
            #yield self._pixel_augmentation(input_data), self._crop_flip_pad(input_data)
            yield self._affine_transformation(self._pixel_augmentation(input_data)), self._crop_flip_pad(input_data)


import time,glob

images = glob.glob('/Users/longshangbang/Documents/Total-Text-Dataset-master/Images/Test/*jpg')


shuffle(images)

image=cv2.imread(images[0])

DA=DataAugmentor()

x=input('start to demo:')
input_={
    'img':image,
    'contour':[np.cast['int32'](np.random.uniform(0,1,(2,1,2))*min(image.shape[:2]))]+[np.array([[[10,100]],[[100,200]]])],
    'type':'tl',
    'is_text_cnts':'False'
}
image_output=DA.augment(input_)

while x=='':
    start=time.time()
    image_,crop_point_starting=next(image_output)
    print(time.time()-start)
    img = image_['img']
    for i in range(len(image_['contour'])):
        for point in range(image_['contour'][i].shape[0]):
            img[image_['contour'][i][point,0,0]-5:image_['contour'][i][point,0,0]+5,
            image_['contour'][i][point,0,1]-5:image_['contour'][i][point,0,1]+5,:]=255
    #print(crop_point_starting[0]-256-10,crop_point_starting[0]-256+10, crop_point_starting[1]-256-10,crop_point_starting[1]-256+10)
    #if crop_point_starting[0]>=266 and crop_point_starting[1]+266<min(img.shape[:2]):
    #    img[crop_point_starting[0]-256-10:crop_point_starting[0]-256+10, crop_point_starting[1]-256-10:crop_point_starting[1]-256+10, :] = 255
    #    img[crop_point_starting[0] + 256-10:crop_point_starting[0]+256+10, crop_point_starting[1] + 256-10:crop_point_starting[1]+256+10, :] = 255
    cv2.imshow('show',np.transpose(img,axes=[1,0,2]))
    cv2.waitKey(1)
    #time.sleep(0.3)
    x = input('enter to see next:')
