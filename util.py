#   to_be_determined
#   please refer to https://github.com/aleju/imgaug
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from random import shuffle, randint, choice, random
import cv2
import copy
import time
import glob


class DataAugmentor(object):
    """
    all data_augmentation defined below should take input as :
    {'img_name':str,   original_name
        'img':np.uint8,
        'contour':List[np.array(the contour of each text instance), (n,1,2)], ---> np.array(num_TI, num_point, 1, 2),
        'center_point':[(x1,y1),(x2,y2)]
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

    it takes around 0.25~0.3s to generate one augmented image on average.
    50% are processed with affine transformation

    """
    def __init__(self):
        self.add_augmentation_list = [
            iaa.Add((-50, 50), per_channel=True),
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
            iaa.GaussianBlur((2, 3)),
            iaa.AverageBlur((2, 3)),
            iaa.MedianBlur((3, 5)),
            iaa.BilateralBlur((2, 3))
        ]

        self.noise_augmentation_list = [
            iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=True),
            iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=False),
            iaa.Dropout((0.05, 0.15), per_channel=False),
            iaa.Dropout((0.05, 0.15), per_channel=True),
            iaa.CoarseDropout((0.05, 0.15), size_percent=(0.5, 0.7))
            # iaa.SaltAndPepper((0.05, 0.15), per_channel=True),
            # iaa.SaltAndPepper((0.05, 0.15), per_channel=False)
        ]

        self.other_augmentation_list = [
            iaa.Sharpen((0.9, 0.11), (0.8, 1.2)),
            iaa.Emboss((0.9, 0.11), (0.3, 1.6)),
            iaa.EdgeDetect((0, 0.4)),
            iaa.Grayscale((0, 1))
        ]

    def _get_seq(self, affine=False):
        if affine:
            affine = [
                iaa.Affine(rotate=360*np.random.rand()-180),
                iaa.Affine(shear=80*np.sin(np.random.rand()*np.pi/2)-40),
                iaa.Affine(scale={"x": 0.4*np.random.rand()+0.8, "y": 0.4*np.random.rand()+0.8})
            ]
            return iaa.Sequential([choice(affine)])

        shuffle(self.add_augmentation_list)
        add_augmentation_list = self.add_augmentation_list[:randint(0, 1)]
        shuffle(self.other_augmentation_list)
        other_augmentation_list = self.other_augmentation_list[:randint(0, 1)]
        shuffle(self.blur_augmentation_list)
        blur_augmentation_list = self.blur_augmentation_list[:randint(0, 1)]
        shuffle(self.noise_augmentation_list)
        noise_augmentation_list = self.noise_augmentation_list[:randint(0, 1)]

        final_list = add_augmentation_list + noise_augmentation_list + blur_augmentation_list + other_augmentation_list

        if len(final_list) == 0:
            final_list.append(iaa.Noop())

        return iaa.Sequential(final_list)

    @staticmethod
    def _key_points(image_shape, point_list):
        """
        feed cnt and return ia.KeypointsOnImage object
        :param point_list: np.array size=(n,1,2)
               image_shape
        :return:
        """
        keypoint_list = []
        for i in range(point_list.shape[0]):
            keypoint_list.append(ia.Keypoint(x=point_list[i, 0, 0], y=point_list[i, 0, 1]))
        return ia.KeypointsOnImage(keypoint_list,
                                   shape=ia.quokka(size=image_shape[:2]))

    @staticmethod
    def _resize(input_data):
        """
        resize image under 512P to 512P
        :param input_data:
        :return:
        return scaled img
        """
        rate = 1
        if input_data['img'].shape[0] > input_data['img'].shape[1]:
            if True:  # input_data['img'].shape[1] < 512:
                rate = 512/input_data['img'].shape[1]
                seq = iaa.Sequential([
                    iaa.Scale({'height': "keep-aspect-ratio", 'width': 512}, 'cubic')
                ])
                input_data['img'] = seq.augment_image(input_data['img'])
        else:
            if True:  # input_data['img'].shape[0] < 512:
                rate = 512/input_data['img'].shape[0]
                seq = iaa.Sequential([
                    iaa.Scale({'height': 512, 'width': "keep-aspect-ratio"}, 'cubic')
                ])
                input_data['img'] = seq.augment_image(input_data['img'])

        input_data['contour'] = [np.cast['int32'](contour*rate) for contour in input_data['contour']]
        input_data['center_point'] = [(np.cast['int32'](point[0] * rate),
                                       np.cast['int32'](point[1] * rate)) for point in input_data['center_point']]
        return input_data

    def _pad(self, input_data):
        h = input_data['img'].shape[0]
        w = input_data['img'].shape[1]
        max_size = max([int(np.sqrt(np.power(h, 2) + np.power(w, 2))),
                        int(w+h*np.cos(11/36))
                        ]) + 5

        up = (max_size - h) // 2
        down = max_size - up - h
        left = (max_size - w) // 2
        right = max_size - left - w

        input_data['img'] = np.pad(input_data['img'], ((up, down), (left, right), (0, 0)), mode='constant')

        input_data['contour'] = list(
            map(lambda x: np.stack([x[:, :, 0] + up, x[:, :, 1] + left], axis=-1),  # x: np.array(n,1,2)
                input_data['contour']))

        input_data['center_point'] = list(
            map(lambda x: (x[0]+up, x[1]+left),
                input_data['center_point']))
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
        return input_data

    def _affine_transformation(self, inputs, trans_rate=0.5):
        """
        affine types include:
            1. scaling
            2. rotate
            3. shear
            4. aspect ratios
        doing affine transformation
        :return:
        """
        input_data = copy.deepcopy(inputs)

        if random() > 1-trans_rate:
            transformer = self._get_seq(affine=True)
            input_data['img'] = transformer.augment_image(input_data['img'])
            for p, cnt in enumerate(input_data['contour']):
                input_data['contour'][p] = transformer.augment_keypoints([
                    self._key_points(image_shape=input_data['img'].shape, point_list=cnt)
                ])[0]
                input_data['contour'][p] = [np.array([[int(keypoints.y),
                                                       int(keypoints.x)]])[:, ::-1]
                                            for keypoints in input_data['contour'][p].keypoints]
                input_data['contour'][p] = np.stack(input_data['contour'][p], axis=0)

            input_data['center_point'][0] = \
                transformer.augment_keypoints([
                    self._key_points(image_shape=input_data['img'].shape,
                                     point_list=np.array(
                                         [[list(input_data['center_point'])[0]]]
                                     ))
                ])[0].keypoints[0]
            input_data['center_point'][0] = (int(input_data['center_point'][0].y),
                                             int(input_data['center_point'][0].x))[::-1]
            input_data['center_point'][1] = transformer.augment_keypoints([
                    self._key_points(image_shape=input_data['img'].shape,
                                     point_list=np.array(
                                         [[list(input_data['center_point'])[1]]]
                                     ))

                ])[0].keypoints[0]
            input_data['center_point'][1] = (int(input_data['center_point'][1].y),
                                             int(input_data['center_point'][1].x))[::-1]

        input_data['img'] = np.transpose(input_data['img'], axes=[1, 0, 2])  # ？？？

        return input_data

    def _crop_flip_pad(self, input_data):
        """
        flip, pad, crop, as final states.
        :param input_data:
        :return:
        """
        center1 = input_data['center_point'][0]
        center2 = input_data['center_point'][1]
        p = random()
        x = center1[0]+int(p*(center2[0]-center1[0]))
        y = center1[1]+int(p*(center2[1]-center1[1]))
        return x, y

    def augment(self, input_, augment_rate=100, trans_rate=0.5):
        """

        :param input_data:
               Dict{'img_name':str,   original_name
                'img':np.uint8,
                'contour':List[the contour of each text instance],
                'type': 'char' or 'tl',
                'is_text_cnts':if this is synthetext or not}
        :param augment_rate
        :param trans_rate
        :return:
        Dict{'img_name':str,   original_name
            'img':np.uint8,
            'contour':List[the contour of each text instance],
            'type': 'char' or 'tl',
            'flag':if this is synthetext or not,
            'left_top': tuple (x, y), x is row, y is col, please be careful about the order,
                 'right_bottom': tuple (x, y), x is row, y is col}
        """
        input_data=copy.deepcopy(input_)
        if input_data['img'].shape[0] < input_data['img'].shape[1]:
            input_data['center_point'] = [(input_data['img'].shape[0] // 2, input_data['img'].shape[0] // 2),
                                      (input_data['img'].shape[0] // 2, input_data['img'].shape[1] - input_data['img'].shape[0] // 2)]
        else:
            input_data['center_point'] = [(input_data['img'].shape[1] // 2, input_data['img'].shape[1] // 2),
                                      (input_data['img'].shape[0] - input_data['img'].shape[1] // 2, input_data['img'].shape[1] // 2)]

        input_data = self._resize(input_data)

        if (not input_data['is_text_cnts']) or (random() < (1 / augment_rate)):
            center_point = self._crop_flip_pad(input_data)
            return input_data, center_point
        else:
            input_data = self._pad(input_data)
            input_data['img'] = np.transpose(input_data['img'], axes=[1, 0, 2])  # ？？？
            #for i in range(augment_rate):
            transformed = self._affine_transformation(self._pixel_augmentation(input_data), trans_rate=trans_rate)
            center_point = self._crop_flip_pad(transformed)
            return transformed, center_point

    @staticmethod
    def demo(input_data, crop_point_starting):
        """
        show the image and the key points
        :param input_data:
        :param crop_point_starting:
        :return:
        """
        img = copy.deepcopy(input_data['img'])
        img[crop_point_starting[0] - 256:crop_point_starting[0] + 256,
            crop_point_starting[1] - 256:crop_point_starting[1] + 256, :] += 20

        for i in range(len(input_data['contour'])):
            for point in range(input_data['contour'][i].shape[0]):
                img[input_data['contour'][i][point, 0, 0] - 5:input_data['contour'][i][point, 0, 0] + 5,
                    input_data['contour'][i][point, 0, 1] - 5:input_data['contour'][i][point, 0, 1] + 5, :] \
                    = (0, 255, 255)

        img[input_data['center_point'][0][0]-5:input_data['center_point'][0][0]+5,
            input_data['center_point'][0][1]-5:input_data['center_point'][0][1]+5, :] = (255, 255, 0)
        img[input_data['center_point'][1][0]-5:input_data['center_point'][1][0]+5,
            input_data['center_point'][1][1]-5:input_data['center_point'][1][1]+5, :] = (255, 255, 0)
        img[crop_point_starting[0]-10:crop_point_starting[0]+10,
            crop_point_starting[1]-10:crop_point_starting[1]+10, :] = (255, 0, 255)
        cv2.imshow('show', img)  # np.transpose(img, axes=[1, 0, 2]))
        cv2.waitKey(1)


if __name__ == '__main__':
    images = glob.glob('/Users/longshangbang/Documents/Total-Text-Dataset-master/Images/Test/*jpg')
    shuffle(images)
    image = cv2.imread(images[0])

    DA = DataAugmentor()
    input_ = {
        'img': image,
        'contour': [np.cast['int32'](np.random.uniform(0, 1, (10, 1, 2))*min(image.shape[:2])) for _ in range(1)],
        'type': 'tl',
        'is_text_cnts': 'False'
    }

    #image_output = DA.augment(input_)
    #DA.demo(input_, (0, 0))
    x = input('enter to see demo:')
    total = 0
    i_ = 0
    while i_ < 50:
        start = time.time()
        image, crop_point_starting = DA.augment(input_)
        total += time.time()-start
        DA.demo(image, crop_point_starting)
        i_ += 1
        #x=input()
    print(total/i_)
