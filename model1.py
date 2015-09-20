#coding=gbk
import zipfile
import cStringIO as sio
import scipy.misc as misc
import numpy as np
import os

def s2o(s):
    return sio.StringIO(s)

dataset = zipfile.ZipFile('glasses.zip','r')
#print dataset.namelist()
noglasslist = [i for i in dataset.namelist() if i.startswith('glasses/alignedNoneGlasses') and i.endswith('bmp')]
glasslist = [i for i in dataset.namelist() if i.startswith('glasses/alignedGlasses') and i.endswith('bmp')]
#print noglasslist
#print glasslist
def liner():
    "提取眼睛行（带眼镜）"
    #import os
    #try: os.mkdir('glassline')
    #except: pass
    import PIL.Image
    glassmodel = np.empty((len(glasslist),(70-25)*(90-0)),np.uint8)
    idx = 0
    glassorig = np.empty((len(glasslist),105*90),np.uint8)
    for i in glasslist:
        img = PIL.Image.open(s2o(dataset.read(i)))
        glassorig[idx] = misc.fromimage(img).flatten()
        img=img.crop((0,25,90,70))
        glassmodel[idx] = misc.fromimage(img).flatten()
        #img.save('glassline\\'+i.split('/')[-1])
        #print i
        idx+=1
    #print glassmodel.shape
    np.save('glassorig.npy', glassorig)
    np.save('glassline.npy',glassmodel)

    nglassmodel = np.empty((len(noglasslist),(70-25)*(90-0)),np.uint8)
    idx = 0
    for i in noglasslist:
        img = PIL.Image.open(s2o(dataset.read(i)))
        img=img.crop((0,25,90,70))
        nglassmodel[idx] = misc.fromimage(img).flatten()
        #img.save('glassline\\'+i.split('/')[-1])
        print i
        idx+=1
    print nglassmodel.shape
    np.save('nglassline.npy',nglassmodel)

def sccodedirect():
    "得到不带眼镜的Sparse Coding结果"
    nglassmodel = np.load('nglassline.npy').astype('f')
    import cPickle
    learning = cPickle.load(file('sparsedirect','rb'))
    #from sklearn.decomposition import MiniBatchDictionaryLearning
    #learning = MiniBatchDictionaryLearning(1000,verbose=True)
    for i in range(20):
        print "Iteration",i
        learning.fit(nglassmodel)
        import cPickle
        cPickle.dump(learning,file('sparsedirect','wb'),-1)

def scdecomp():
    "对带眼镜照片进行分解，得到还原表示以及眼镜区域"
    glassmodel = np.load('glassline.npy').astype('f')
    import cPickle
    sparsedirect = cPickle.load(file('sparsedirect','rb'))
    sparsedirect.set_params(transform_algorithm='lasso_lars')
    dout = sparsedirect.transform(glassmodel)
    drec = dout.dot(sparsedirect.components_)
    np.save('glassdecomp.npy',drec)
    np.save('glassdiff.npy',glassmodel - drec)

def scvisdecomp():
    "可视化结果"
    from layerbase import DrawPatch
    import cPickle
    sparsedirect = cPickle.load(file('sparsedirect','rb'))
    drec = sparsedirect.components_.reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('dict.jpg')
    drec = np.load('glassline.npy').astype('f').reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('src.jpg')
    drec = np.load('glassdecomp.npy').reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('decomp.jpg')
    drec = np.abs(np.load('glassdiff.npy')).reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('diff.jpg')

def cutratio(ratio,imgid,sparsedirect=None, glassmodel=None):
    "使用卡阈值标记损坏区域的方法迭代SparseCoding去除眼镜"
    if sparsedirect==None:
        import cPickle
        sparsedirect = cPickle.load(file('sparsedirect','rb'))
        sparsedirect.set_params(transform_algorithm='lasso_lars')
    modelval = sparsedirect.components_
    if glassmodel==None:
        glassmodel = np.load('glassline.npy').astype('f')
    sample = glassmodel[imgid:imgid+1]

    #初始还原损坏区域
    maskarea = np.where(misc.imread('glasses.bmp')[:,:,0]>10,0,1).flatten()
    #rec = sparsedirect.transform(sample).dot(modelval)
    #diff = np.abs(sample - rec)[0]
    #diffmax = np.max(diff)
    #maskarea = np.where(diff>diffmax*ratio,0,1)
    
    itx = 1
    while True:
        newmodel = np.copy(modelval)
        sparsedirect.components_ = newmodel * maskarea[np.newaxis,:]
        newsample = np.copy(sample)
        newsample = newsample * maskarea[np.newaxis,:]

        rec = sparsedirect.transform(newsample).dot(modelval)
        diff = np.abs(sample - rec)[0]
        diffmax = np.max(diff)
        
        newmaskarea = np.where(diff>diffmax*ratio,255,0).reshape((70-25,90-0)).astype(np.uint8)
        misc.toimage((diff*255/np.max(diff)).reshape((70-25,90-0)).astype(np.uint8)).convert('RGB').save('input.bmp')
        os.popen('CollectionFlow.exe','r').read()
        newmaskarea = np.where(misc.imread('output.bmp')>30,0,1).flatten()
        misc.toimage((newmaskarea*255).reshape((70-25,90-0)).astype(np.uint8)).save('cuts.bmp')
        maskdiff = np.sum(np.abs(maskarea - newmaskarea))
        print "ITERATION",itx,"DIFF",maskdiff
        itx += 1
        maskarea = newmaskarea
        if maskdiff <= 40 or itx>10:
            print "Converge"
            break
    sparsedirect.components_ = modelval
    return rec, diff, maskarea

def cutratio_all(ratio):
    import cPickle
    sparsedirect = cPickle.load(file('sparsedirect','rb'))
    sparsedirect.set_params(transform_algorithm='lasso_lars')
    glassmodel = np.load('glassline.npy').astype('f')
    glassmodel = glassmodel[0:4]

    recall = np.empty_like(glassmodel)
    diffall = np.empty_like(glassmodel)
    maskall = np.empty_like(glassmodel)

    for idx in range(glassmodel.shape[0]):
        print idx,glassmodel.shape[0]
        recall[idx],diffall[idx],maskall[idx] = cutratio(ratio, idx, sparsedirect, glassmodel)

    np.save('cut_rec.npy',recall)
    np.save('cut_diff.npy',diffall)
    np.save('cut_mask.npy',maskall)

def showcut():
    recall = np.load('cut_rec.npy')
    diffall = np.load('cut_diff.npy')
    maskall = np.load('cut_mask.npy')
    from layerbase import DrawPatch
    drec = recall.reshape((-1,1,70-25,90-0))
    drec2 = np.load('glassline.npy').astype('f').reshape((-1,1,70-25,90-0))[0:4]
    drecall = np.zeros((drec.shape[0]+drec2.shape[0],1,70-25,90-0),'f')
    drecall[::2]=drec
    drecall[1::2]=drec2
    misc.toimage(DrawPatch(drecall)).save('ipair.jpg')
    drec = diffall.reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('idiff.jpg')
    drec = maskall.reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('imask.jpg')

if __name__=="__main__":
    print "Make model 1"
    liner()
    #sccodedirect()
    #scdecomp()
    #scvisdecomp()
    #cutratio_all(0.3)
    #showcut()
