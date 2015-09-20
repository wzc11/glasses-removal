#coding=gbk
import zipfile
import cStringIO as sio
import scipy.misc as misc
import numpy as np

def s2o(s):
    return sio.StringIO(s)

dataset = zipfile.ZipFile('glasses.zip','r')
noglasslist = [i for i in dataset.namelist() if i.startswith('alignedNoneGlasses') and i.endswith('bmp')]
glasslist = [i for i in dataset.namelist() if i.startswith('alignedGlasses') and i.endswith('bmp')]

def buildmodel2():
    "生成有眼镜-无眼镜pair模型"
    modelrec = np.load('cut_rec.npy')
    modelglass = np.load('glassline.npy')[:modelrec.shape[0]]

    linkedmodel = np.empty((modelrec.shape[0],modelrec.shape[1]+modelglass.shape[1]),'f')
    linkedmodel[:,:modelrec.shape[1]]=modelrec
    linkedmodel[:,modelrec.shape[1]:]=modelglass

    #Train
    from sklearn.decomposition import MiniBatchDictionaryLearning
    learning = MiniBatchDictionaryLearning(500,verbose=True)
    learning.fit(linkedmodel)
    import cPickle
    cPickle.dump(learning,file('sparselinked','wb'),-1)

def pairdump():
    "可视化模型列"
    from layerbase import DrawPatch
    import cPickle
    sparsedirect = cPickle.load(file('sparselinked','rb'))
    drec = sparsedirect.components_.reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('dictpair.jpg')

if __name__=="__main__":
    buildmodel2()
    pairdump()
