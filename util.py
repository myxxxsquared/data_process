#   to_be_determined
#   please refer to https://github.com/aleju/imgaug

class data_augmentor(object):
    """
    all data_augmentation defined below should take input as :
    {'img_name':str,   original_name
        'img':np.uint8,
        'contour':List[the contour of each text instance],
        'type': 'char' or 'tl',
        'flag':if this is synthetext or not}

    while returning the same format.

    valuable data augmentation types:
        crop, pad, flip, invert, add(channel), add to hue and saturation, multiply,
        gaussian blur, average blur, median blur

    """
    def __init__(self,*args,**kw):
        pass

    def resize(self,*args,**kw):
        pass

    def cropping(self,*args,**kw):
        pass

    def flipping(self,*args,**kw):
        pass

    def affine_transformation(self,*args,**kw):
        pass