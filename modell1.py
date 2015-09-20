#coding=gbk
from scipy.optimize import minimize, check_grad

#coding=gbk
import zipfile
import cStringIO as sio
import scipy.misc as misc
import numpy as np
import numpy.random as npr
import os
try: from gnumpy import dot
except: from numpy import dot

def rpca_learn(datamat,dicts=500,learnstep=1000,iterperstep=50,stepdict=None,stepcode=None,l1_recon=1.0,l1_code=1.0,l2_dict=1.0,eps=1e-10):
    "Robust PCA算法字典学习(离线)"
    dictshape = (dicts,datamat.shape[1])
    if stepdict==None:
        stepdict = npr.ranf(dictshape)
    codeshape = (datamat.shape[0],dicts)
    CODELEN = np.prod(codeshape)
    if stepcode==None:
        stepcode = np.zeros(codeshape)

    def fmin(code_dict):
        code = code_dict[:CODELEN].reshape(codeshape)
        idict = code_dict[CODELEN:].reshape(dictshape)
        recons = dot(code,idict)
        error = recons - datamat

        loss_recon = np.sqrt(error * error + eps)
        loss_code = np.sqrt(code * code + eps)
        loss_dict = idict * idict
        loss = l1_recon * loss_recon.sum() + l1_code * loss_code.sum() + l2_dict * loss_dict.sum()

        grad_recon = l1_recon * error / loss_recon
        grad_code = l1_code * code / loss_code
        grad_dict = l2_dict * idict * 2

        grad_code_2 = dot(grad_recon,idict.T)
        grad_dict_2 = dot(code.T,grad_recon)

        grad_code = grad_code + grad_code_2
        grad_dict = grad_dict + grad_dict_2
        print loss_recon.sum(), loss_code.sum(), loss_dict.sum(), loss
        return loss, np.concatenate((grad_code.flatten(),grad_dict.flatten()))

    for itx in range(learnstep):
        print "ITERATION",itx
        mres = minimize(fmin, np.concatenate((stepcode.flatten(),stepdict.flatten())), method='L-BFGS-B', jac=True, options={'maxiter':iterperstep, 'disp':False})
        stepcode = mres.x[:CODELEN].reshape(codeshape)
        stepdict = mres.x[CODELEN:].reshape(dictshape)

        yield stepdict, stepcode

def rpca_noglass():
    "用RPCA学习不戴眼镜的字典"
    dic = np.load('nglassline.npy')
    dicflat = dic.reshape((dic.shape[0],-1))
    import os
    if os.path.exists('eyemodel.npz'):
        partialmodel = np.load('eyemodel.npz')
    else:
        partialmodel = {'components':None, 'nglasscodes':None}

    from layerbase import DrawPatch
    idx=0
    for sdict, scode in rpca_learn(dicflat,stepdict=partialmodel['components'],stepcode=partialmodel['nglasscodes']):
        np.savez('eyemodel.npz', components=sdict, nglasscodes=scode)
        misc.toimage(DrawPatch(sdict.reshape((-1,1,45,90)),False,'bgy')).save('dictrpca_%s.png'%idx)
        print "SAVED"
        idx += 1

def rpca_recons(sparsecode,vec,l1_recon=1.0,l1_code=1.0,eps=1e-10):
    "Robust PCA算法还原人脸效果"
    basis = sparsecode.components_
    orgbasis = basis
    basis = np.copy(basis)
    orgvec = vec
    vec = np.copy(vec)

    origcode = sparsecode.transform(vec.reshape((1,-1))).flatten()
    
    def fmin(code):
        recons = code.reshape((1,-1)).dot(basis).flatten()
        errors = recons - vec

        loss_recon = np.sqrt(errors*errors + eps)
        loss_code = np.sqrt(code*code + eps)
        loss = l1_recon * loss_recon.sum() + l1_code * loss_code.sum()

        grad_recon = l1_recon * errors / loss_recon
        grad_code = l1_code * code / loss_code

        grad_code_2 = grad_recon.reshape((1,-1)).dot(basis.T).flatten()
        grad = grad_code + grad_code_2
        #print '%s\t%s\t%s'%(loss_recon.sum(),loss_code.sum(),loss)
        return loss, grad
    #print check_grad(lambda x:fmin(x)[0],lambda x:fmin(x)[1], origcode)
    #import sys
    #sys.exit(0)
    maskarea = np.ones_like(vec)

    itx = 0
    from scipy import ndimage
    from scipy.ndimage.filters import gaussian_filter
    while True:
        origcode = minimize(fmin, origcode, method='L-BFGS-B', jac=True, options={'maxiter':200, 'disp':False}).x
        result = origcode.reshape((1,-1)).dot(orgbasis).flatten().clip(0,1e100)
        try:
            misc.toimage((result*255/np.max(result)).reshape((70-25,90-0)).astype(np.uint8)).convert('RGB').save('partresult.bmp')
        except:
            pass
        diff = np.abs(orgvec - result)
        diffimg = (diff*255/np.max(diff)).reshape((70-25,90-0))
        dx = ndimage.sobel(diffimg, 0)
        dy = ndimage.sobel(diffimg, 1)
        diffimg = np.sqrt(dx*dx+dy*dy+diffimg*diffimg*20)
        diffimg = diffimg*255/np.max(diffimg)
        while True:
            try:
                misc.toimage(diffimg.astype(np.uint8)).convert('RGB').save('input.bmp')
                break
            except:
                pass
        os.popen('CollectionFlow.exe','r').read()
        out = misc.imread('output.bmp')
        out = gaussian_filter(out,0.5)
        newmaskarea = np.where(out>40,0,1).flatten()
        try: misc.toimage((newmaskarea*255).reshape((70-25,90-0)).astype(np.uint8)).save('cuts.bmp')
        except: pass
        maskdiff = np.sum(np.abs(maskarea - newmaskarea))
        print "ITERATION",itx,"DIFF",maskdiff
        itx += 1
        maskarea = newmaskarea
        if (maskdiff<=20 and itx > 10) or itx>20:
            print "Converge"
            break
        vec = orgvec * maskarea
        basis = orgbasis * maskarea[np.newaxis,:]

    return result, np.abs(vec - result)

def rpca_all():
    "用RPCA在所有眼镜人脸上进行迭代"
    import cPickle
    sparsedirect = cPickle.load(file('sparsedirect','rb'))
    sparsedirect.components_ = np.load('eyemodel.npz')['components']

    glassmodel = np.load('glassline.npy').astype('f')
    glassmodel = glassmodel

    recall = np.empty_like(glassmodel)
    diffall = np.empty_like(glassmodel)

    for idx in range(glassmodel.shape[0]):
        print idx,glassmodel.shape[0]
        recall[idx],diffall[idx] = rpca_recons(sparsedirect, glassmodel[idx])

    np.save('rpca_rec.npy',recall)
    np.save('rpca_diff.npy',diffall)

def showrpca():
    recall = np.load('rpca_rec.npy')
    diffall = np.load('rpca_diff.npy')
    from bigrec import bigrec
    from layerbase import DrawPatch
    drec = recall.reshape((-1,1,70-25,90-0))
    drec2 = np.load('glassline.npy').astype('f').reshape((-1,1,70-25,90-0))
    glassorig = np.load('glassorig.npy').reshape((-1,105,90))
    glassall = np.copy(glassorig)[:drec.shape[0]]
    drecall = np.zeros((drec.shape[0]+drec2.shape[0],1,70-25,90-0),'f')
    drecall[::2]=drec
    drecall[1::2]=drec2
    for i in range(drec.shape[0]):
        glassall[i] = bigrec(drec[i,0], glassall[i])
    misc.toimage(DrawPatch(glassall.reshape((-1,1,105,90)))).save('glasses.jpg')
    
    drecall = np.zeros((drec.shape[0]+drec.shape[0],105,90),'f')
    drecall[::2]=glassall
    drecall[1::2]=glassorig
    misc.toimage(DrawPatch(drecall.reshape((-1,1,105,90)))).save('recoverypair.jpg')
    
    #return
    #misc.toimage(DrawPatch(drecall)).save('rpcapair.jpg')
    drec = diffall.reshape((-1,1,70-25,90-0))
    misc.toimage(DrawPatch(drec)).save('rpcadiff.jpg')

def makedata():
    "生成用于神经网络训练的数据"
    diffall = np.load('rpca_diff.npy').reshape((-1,1,70-25,90-0)).astype('f')
    glassmodel = np.load('glassline.npy').reshape((-1,1,70-25,90-0)).astype('f')
    print diffall.shape,glassmodel.shape
    from scipy import ndimage
    from scipy.ndimage.filters import gaussian_filter
    for i in range(diffall.shape[0]):
        print i
        sdiff = diffall[i,0]
        sdiff = (sdiff - np.min(sdiff))/(np.max(sdiff) - np.min(sdiff))
        misc.toimage((sdiff*255).astype(np.uint8)).convert('RGB').save('input.bmp')
        os.popen('CollectionFlow.exe','r').read()
        out = misc.imread('output.bmp')
        out = gaussian_filter(out,0.5)
        newmaskarea = np.where(out>40,255,0)
        diffall[i,0]=newmaskarea
    np.savez('glassdata.npz',input=glassmodel,output=diffall)

def showdata():
    "绘出数据"
    data = np.load('glassdata2.npz')
    from layerbase import DrawPatch
    misc.toimage(DrawPatch(data['input'])).save('datainput.png')
    misc.toimage(DrawPatch(data['output'])).save('dataoutput.png')

def largestpart():
    "找出数据中最大的部分"
    data = np.load('glassdata.npz')
    import itertools
    adjmat = [(i,j) for i,j in itertools.product(range(-3,4),range(-3,4)) if (i!=0 or j!=0)]
    print adjmat
    from collections import deque
    glassmodel = data['output']
    for i in range(glassmodel.shape[0]):
        print i
        adjcount=[]
        inp = np.where(glassmodel[i,0]>200,-2,-1)
        pos = deque()
        if 1:
            for py in range(inp.shape[0]):
                for px in range(inp.shape[1]):
                    if inp[py,px]!=-2: continue
                    pos.appendleft((py,px))
                    adjcount.append(0)
                    fp = len(adjcount)
                    while len(pos)>0:
                        py2,px2 = pos.pop()
                        if inp[py2,px2]!=-2: continue
                        inp[py2,px2] = fp
                        for dy,dx in adjmat:
                            y2=py2+dy
                            x2=px2+dx
                            if x2>=0 and x2<inp.shape[1] and y2>=0 and y2<inp.shape[0]:
                                pos.appendleft((y2,x2))
                                adjcount[fp-1] += 1
        #print adjcount
        keepval = np.argmax(adjcount)
        glassmodel[i,0]=np.where(inp==keepval+1,255,0)
    #data['output']=glassmodel
    np.savez('glassdata2.npz',input=data['input'],output=glassmodel)

def maskedmedfilt(img,mask,ri,rj):
    "中值滤波"
    import numpy.ma as ma
    iout = np.empty_like(img)
    print img.shape,mask.shape
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imin = (i-ri) if (i-ri>0) else 0
            imax = (i+ri) if (i+ri<img.shape[0]) else (img.shape[0]-1)
            jmin = (j-rj) if (j-rj>0) else 0
            jmax = (j+rj) if (j+rj<img.shape[1]) else (img.shape[1]-1)
            pimg = img[imin:imax+1,jmin:jmax+1].flatten()
            pmask = mask[imin:imax+1,jmin:jmax+1].flatten().astype('b')
            masked = ma.array(pimg,mask=pmask).compressed()
            masked.sort()
            iout[i,j]=masked[masked.shape[0]/2]
    return iout

def domedmask():
    diff = np.load('rpca_diff.npy').reshape((-1,70-25,90-0))
    diffmask = np.where(diff>50,255,0)
    from bigrec import bigrec
    maskall = np.zeros((diff.shape[0],105,90),np.int0)
    glassorig = np.load('glassorig.npy').reshape((-1,105,90))
    noglass = np.zeros_like(glassorig)
    for i in range(diff.shape[0]):
        print i
        maskall[i] = bigrec(diffmask[i], maskall[i])
        medresult = maskedmedfilt(glassorig[i], maskall[i], 6,4)
        noglass[i] = np.where(maskall[i], medresult, glassorig[i])
    np.save('medfilt_noglass.npy',noglass)
    from layerbase import DrawPatch
    misc.toimage(DrawPatch(noglass.reshape((-1,1,105,90)))).save('mednoglass.png')
    misc.toimage(DrawPatch(maskall.reshape((-1,1,105,90)))).save('medmask.png')
    
if __name__=="__main__":
    print "Model RPCA"
    #rpca_noglass()
    rpca_all()
    showrpca()
    #makedata()
    #largestpart()
    #showdata()
    #domedmask()

