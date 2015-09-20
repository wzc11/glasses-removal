#coding=gbk
import numpy as np
import scipy.misc as misc

def showtransform():
    "显示神经网络转换结果"
    trans = np.load('transform.npy')[:,0]
    glassmodel = np.load('glassline.npy').astype('f').reshape((-1,45,90))
    glassmodel /= np.max(glassmodel)
    trans = np.insert(trans,2,glassmodel,axis=1)
    p = trans[:,0].reshape((trans.shape[0],-1))
    trans[:,0] -= ((p.max(axis=1)+p.mean(axis=1))*0.5).reshape((-1,1,1))
    trans[:,0] = np.where(trans[:,0]>0, trans[:,0],0)
    from layerbase import DrawPatch
    misc.toimage(DrawPatch(trans,False,'bgy')).save('nntrans.png')

if __name__=="__main__":
    showtransform()
