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

def SynthText_loader(start_point,end_point):
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