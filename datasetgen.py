#coding=gbk
import sys
import os
import numpy as np
import scipy.misc as misc

def liner(d):
    "提取眼睛行（带眼镜）"
    #import os
    #try: os.mkdir('glassline')
    #except: pass
    import PIL.Image
    glasslist = [i for i in os.listdir(d) if i.endswith('bmp')]
    glassmodel = np.empty((len(glasslist),(70-25)*(90-0)),np.uint8)
    idx = 0
    glassorig = np.empty((len(glasslist),105*90),np.uint8)
    for i in glasslist:
        print i
        img = PIL.Image.open(os.path.join(d,i))
        glassorig[idx] = misc.fromimage(img).flatten()
        img=img.crop((0,25,90,70)).convert('L')
        glassmodel[idx] = misc.fromimage(img).flatten()
        #img.save('glassline\\'+i.split('/')[-1])
        #print i
        idx+=1
    #print glassmodel.shape
    np.save('glassorig.npy', glassorig)
    np.save('glassline.npy',glassmodel)


if __name__=="__main__":
    directory = sys.argv[1]
    liner(directory)
