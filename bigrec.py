import numpy as np
import scipy.misc as misc
import os

def bigrec(glassline, origimg):
    origimg=np.copy(origimg)
    origimg[25:25+glassline.shape[0]]=glassline.clip(0,255)
    return origimg
    mask = np.zeros((glassline.shape[0],glassline.shape[1]),np.uint8)
    mask[4:-4,4:-4] = 255
    misc.toimage(mask).convert('L').save('blendmask.bmp')
    print glassline.shape,origimg.shape
    misc.toimage(glassline).convert('RGB').save('glassimg.bmp')
    misc.toimage(origimg).convert('RGB').save('background.bmp')

    print os.popen('pb.exe background.bmp glassimg.bmp blendmask.bmp 0 25 false').read()
    return misc.imread('test.bmp')
