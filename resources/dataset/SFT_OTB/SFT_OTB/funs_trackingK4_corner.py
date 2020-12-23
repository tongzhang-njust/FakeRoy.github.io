'''
funs_tracking.py
'''

import numpy as np
import scipy.misc as MISC
#from theano.tensor.signal.feaprojecting import *
#from theano.tensor.signal.feapoolingRT import *
from PIL import Image, ImageFont, ImageDraw
#from fhog import *
import subprocess
import scipy.ndimage as ndimage
from utis import *

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
#from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from scipy import interpolate
import scipy.misc as MISC
from scipy.spatial.distance import cdist
import time
from numpy.linalg import inv

'''
close_logger(logger)
fun_correlation_singlesample(xf,yf,sigma,filter_type)
fun_cos_win(h,w)
get_ctr_shift_trackers(responses,weight_trackers)
get_weights_bmap(batch_bmap_y)
fun_binarize_map(prop)
fun_get_binary_map(win_size,rct_size,cell_size)
fun_get_freq_fea(x,is_cos_window)
fun_get_max_response(x1,x2,x3,x4)
fun_get_patch(x,ctr,sz)
fun_get_patch_warping(img,ctr,tmp_sz,win_sz)
fun_get_peak_map(win_size,rct_size,cell_size,fea_size,isroll)
fun_get_search_window(target_sz,im_sz,padding)
fun_get_search_ctr(rct,factor=1.)
fun_get_strnn_list(srnn_directions)
fun_kernel_multisample(xf,yf,kernel_type,sigma)
fun_draw_rct_on_image(im,fname,rct1,rct2,rct3)
fun_process_binary_image(x)
fun_response(model_w,xf,zf,kernel_type,sigma,gamma)
fun_shift_feas(x,px,directions,step,outx)
fun_vggfea_list2array(xlist,nmh,nmw,interp_type,flag,outx)
fun_w(xf,yf,kernel_type,sigma,gamma)
'''

def convert_contour_to_rect(contour_points):
    ## contour_points: [n,8], [tl, tr, br, bl], [w,h]
    n = contour_points.shape[0]
    rct = np.zeros((n,4), dtype=np.float32) # [tl, width, height], [w,h]
    rct[:,0] = (contour_points[:,0::2]).min(axis=-1)
    rct[:,1] = (contour_points[:,1::2]).min(axis=-1)
    rct[:,2] = (contour_points[:,0::2]).max(axis=-1)
    rct[:,3] = (contour_points[:,1::2]).max(axis=-1)
    rct[:,2] = rct[:,2] - rct[:,0]
    rct[:,3] = rct[:,3] - rct[:,1]
    return rct
    

def get_contour_nscale_minus_offset(ocontour, new_im_offset, ctr_offset, scale, ref_ctr_h, ref_ctr_w):
    ## contour_1frame [w1,h1,w2,h2,...,w4,h4]
    ## new_im_offset: [h,w]
    ## ctr_offset: [w,h]
    ## scale

    new_contour = np.reshape(ocontour, (4,2)) # [w,h]
    new_contour = new_contour - np.reshape(new_im_offset[::-1],(1,2)) + np.reshape(ctr_offset,(1,2)) # [4,2], [w,h]

    w = new_contour[:,0] 
    new_contour[:,0] = (w - ref_ctr_w)*scale + ref_ctr_w
    
    h = new_contour[:,1]
    new_contour[:,1] = (h - ref_ctr_h)*scale + ref_ctr_h

    return np.reshape(new_contour,(8,))
    
    
    

def get_ori_coordinates(pos_h, pos_w, transform, offset):
    ## offset: [h, w] w.r.t. original image
    ## return [w,h] 
    ##
    a0, a1, a2, b0, b1, b2, c0, c1 = np.squeeze(transform)
    x,y = pos_w, pos_h
    k = c0* x + c1* y + 1
    pos_w, pos_h = (a0* x + a1* y + a2) / k, (b0* x + b1* y + b2) / k
    pos_w, pos_h = pos_w + offset[1], pos_h + offset[0]
    return np.asarray([pos_w, pos_h])


def get_scale_factors(nscale, scale_step, im_sz, tgt_sz, win_sz):#(ctr, winsz, nscale, scale_step, im_sz, tgt_sz):
    # note: [w,h]

    assert(nscale % 2 == 1)
    scale_exp = np.arange(-np.floor((nscale-1)/2), np.ceil((nscale+1)/2))
    scaleFactors = scale_step**scale_exp

    min_scale_factor = scale_step**np.ceil(np.log(np.max(5./win_sz))/np.log(scale_step))
    max_scale_factor = scale_step**np.floor(np.log(np.min(im_sz/tgt_sz))/np.log(scale_step))
    
    return scaleFactors, min_scale_factor, max_scale_factor

def get_bounding_boxes_for_nscale(ctr, owin_sz, currentScaleFactor, scaleFactors, padFactor = 1.2):
    # all are [w, h]
    # ctr: sampling center
    # owin_sz: the standard sampling window size, which will be multipied with currentScaleFactor and scaleFactors
    # padFactor: final crop factor
    # return [h,w] style

    ## bounding box
    nscale = len(scaleFactors)
    old_bbx_hw = np.zeros((nscale, 4), dtype=np.float32) # [h1,w1,h2,w2]
    
    def compute_bbx_float(sf):
        win_sz = owin_sz * currentScaleFactor * sf
        h1 = ctr[1]-win_sz[1]/2. #np.int32(np.floor(ctr[1]-win_sz2[1]/2+0.5)) 
        w1 = ctr[0]-win_sz[0]/2. #np.int32(np.floor(ctr[0]-win_sz2[0]/2+0.5)) 
        h2, w2 = h1 + win_sz[1], w1 + win_sz[0]
        return np.asarray([h1,w1,h2,w2]), win_sz

    def compute_bbx_int(sf):
        win_sz = np.floor(owin_sz * currentScaleFactor * sf + 0.5)
        if win_sz[0] < 1:
            win_sz[0] = 2
        if win_sz[1] < 1:
            win_sz[1] = 2
        h1 = np.int32(np.floor(ctr[1]-win_sz2[1]/2+0.5)) # ??
        w1 = np.int32(np.floor(ctr[0]-win_sz2[0]/2+0.5)) # ??
        h2, w2 = h1 + win_sz[1], w1 + win_sz[0]
        return np.asarray([h1,w1,h2,w2]), win_sz

    for ii in range(nscale):
        #old_bbx_hw[ii,:] = compute_bbx_int(scaleFactors[ii]) 
        old_bbx_hw[ii,:],_ = compute_bbx_float(scaleFactors[ii])

    bind = np.zeros(nscale, dtype=np.int32)
 
    ## big sampling size 
    smp_sz =  np.floor(owin_sz * currentScaleFactor * padFactor + 0.5)
    idx = np.mod(smp_sz,2) == 0
    smp_sz[idx] = smp_sz[idx] + 1
    smp_sz_hw = smp_sz[::-1] # [h,w]

    ## 
    smp_offset = ctr[::-1] - np.floor(smp_sz_hw/2) # [h,w] 

    ## new_bbx_hw: ratio [0,1] 
    new_bbx_hw = np.reshape(old_bbx_hw, (-1,2,2)) # [nbbx, 2,2] [h1,w1;h2,w2]
    new_bbx_hw = new_bbx_hw - np.reshape(smp_offset,(1,1,2)) # [nbbx,2,2]
    new_bbx_hw_ratio = new_bbx_hw/(np.reshape(smp_sz_hw,(1,1,2))-1) # [nbbx, 2, 2]
    new_bbx_hw = np.reshape(new_bbx_hw, (-1,4)) # [nbbx, 4]  
    new_bbx_hw_ratio = np.reshape(new_bbx_hw_ratio, (-1,4)) # [nbbx, 4]   
 
    return old_bbx_hw, bind, smp_sz_hw, smp_offset, new_bbx_hw, new_bbx_hw_ratio 


def dur(str=None,clock=[time.time()]):
    if str!=None:
       duration=time.time()-clock[0]
       print ('%s finished.Duration %.6f seconds.'%(str,duration))
    clock[0]=time.time()
	
def fea_pca_tr(x, p, energy, is_mean = True, is_norm = True):
    # x: h*w*d
    h, w, d = x.shape
    assert(d%p ==0)
    d2 = np.int32(d/p)
    ss = 0
    projections = []
    for ii in range(p):
        z = np.reshape(x[:,:,ss:ss+d2],(h*w,d2))
        ss = ss + d2
        proj = pca_tr(z, energy, is_mean, is_norm) 
        projections.append(proj)

    return projections

def fea_pca_te_ms(x, p, projections):
    # x: n*h*w*d
    n, h, w, d = x.shape
    assert(d%p ==0)
    d2 = np.int32(d/p)
    ss = 0
    o = []
    for ii in range(p):
        z = np.reshape(x[:,:,:,ss:ss+d2],(n*h*w,d2))
        ss = ss + d2
        z = pca_te(z, projections[ii])
        o.append(z.reshape((n,h*w,-1)))

    return np.concatenate(o, axis = -1)
        
def fea_pca_te(x, p, projections):
    # x: h*w*d
    h, w, d = x.shape
    assert(d%p ==0)
    d2 = np.int32(d/p)
    ss = 0
    o = []
    for ii in range(p):
        z = np.reshape(x[:,:,ss:ss+d2],(h*w,d2))
        ss = ss + d2
        z = pca_te(z, projections[ii])
        o.append(z)

    return np.concatenate(o, axis = -1)
    #return o
    #return np.reshape(np.concatenate(o, axis = -1),(h,w,-1))

def compute_hw(mx_hh0,mx_ww0,mx_res0):
    msk_idx = np.logical_not(np.isnan(mx_res0))
    mx_hh = np.copy(mx_hh0)
    mx_ww = np.copy(mx_ww0)
    #assert len(msk_idx) == 6
    n = len(msk_idx)
    n2 = np.int32(n/2)
    
    idx = np.where(msk_idx < 0.5)
    mx_hh[idx] = 0
    mx_ww[idx] = 0
    mx_res0[idx] = 0
    
    sgn_h = np.sum(np.sign(mx_hh[n2:]))
    if sgn_h == n-n2:
       idx = np.where(mx_hh[0:n2]<0)
       mx_hh[idx] = 0
    elif sgn_h == n2-n:
       idx = np.where(mx_hh[0:n2]>0)
       mx_hh[idx] = 0

    sgn_w = np.sum(np.sign(mx_ww[n2:]))
    if sgn_w == n-n2:
       idx = np.where(mx_ww[0:n2]<0)
       mx_ww[idx] = 0
    elif sgn_w == n2-n:
       idx = np.where(mx_ww[0:n2]>0)
       mx_ww[idx] = 0
    #idx = np.where(msk_idx > 0.5)
    
    mx_w = np.mean(mx_ww[msk_idx])#[mx_layer]
    mx_h = np.mean(mx_hh[msk_idx])#[mx_layer]
    mxres = np.mean(mx_res0[msk_idx])#[mx_layer]
    return mx_w,mx_h,mxres


def fun_get_patch_warping(img,ctr,tmp_sz,win_sz):
    # [w,h]
    
    img = np.float32(img)
    if len(img.shape) == 3:
        isColor = True
    else:
        isColor = False
    h = img.shape[0]
    w = img.shape[1]

    x = np.arange(1,win_sz[0]+1)-win_sz[0]/2+0.5
    y = np.arange(1,win_sz[1]+1)-win_sz[1]/2
    [x,y] = np.meshgrid(x,y)
    p3 = tmp_sz[0]/win_sz[0]
    #print p3,p3*tmp_sz[1]/win_sz[1]
    yp = ctr[1] + y*(p3*tmp_sz[1]/win_sz[1])-1
    xp = ctr[0] + x*p3-1
    
    #save_mat_file('warping.mat',x,y,xp,yp) #??
    ##
    x0 = np.int32(xp)
    x1 = x0+1
    y0 = np.int32(yp)
    y1 = y0+1

    rx0 = xp-x0
    rx1 = 1-rx0
    ry = yp-y0
    
    ## --
    
    x0_bool = (x0<0)+(x0>w-1)
    x1_bool = (x1<0)+(x1>w-1)
    y0_bool = (y0<0)+(y0>h-1)
    y1_bool = (y1<0)+(y1>h-1)

    x0[x0_bool] = 0
    x1[x1_bool] = 0
    y0[y0_bool] = 0
    y1[y1_bool] = 0
    
    
    if isColor ==True:
        patch = np.zeros((win_sz[1],win_sz[0],3))
        for ii in range(3):
            patch[:,:,ii] = (rx1*img[y0,x0,ii]*(~(y0_bool+x0_bool)) + rx0*img[y0,x1,ii]*(~(y0_bool+x1_bool)))*(1-ry) + \
                    (rx1*img[y1,x0,ii]*(~(y1_bool+x0_bool)) + rx0*img[y1,x1,ii]*(~(y1_bool+x1_bool)))*ry
    else:
        patch = (rx1*img[y0,x0]*(~(y0_bool+x0_bool)) + rx0*img[y0,x1]*(~(y0_bool+x1_bool)))*(1-ry) + \
                       (rx1*img[y1,x0]*(~(y1_bool+x0_bool)) + rx0*img[y1,x1]*(~(y1_bool+x1_bool)))*ry
    
    patch[patch<0] = 0
    patch[patch>255] = 255
    return np.uint8(patch+0.5)



def fun_process_binary_image(x):
    [n,c,h,w] = x.shape
    selem = disk(3)
    y = np.zeros((n,c,h,w),x.dtype)
    for ii in range(n):
        for jj in range(c):
            im = closing(np.asarray(x[ii,jj],dtype=np.uint8), selem)
	    
            im = ndimage.binary_erosion(im, structure=np.ones((3,3))).astype(x.dtype)
            im = np.array(im,dtype=np.uint8)
            ret,thresh = cv2.threshold(im,0.5,1,0)
            contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x0,y0,w0,h0 = cv2.boundingRect(cnt)
            y[ii,jj,y0:y0+h0,x0:x0+w0] = 1

    #x = ndimage.binary_fill_holes(x).astype(x.dtype)
    #x = ndimage.binary_erosion(x, structure=np.ones((4,4)),iterations=2).astype(x.dtype)
    flag = True
    if np.sum(y)<1:
        flag = False
    return y,flag

def compute_kernel_size(target_sz,cell_size,factor=1.):
    kl = np.int32(factor*target_sz/cell_size+0.5)
    if kl[0]%2 == 0:
        kl[0] = kl[0]-1
    if kl[1]%2 == 0:
        kl[1] = kl[1]-1
    return kl

def get_weights_bmap(batch_bmap_y):
    # batch_bmap_y: [n,nclass,h,w]
    [n,nc,h,w] = batch_bmap_y.shape
    x = np.sum(batch_bmap_y,axis=(-2,-1),keepdims=True)
    x = np.asarray(x,dtype=np.float32)
    x = 1/x
    x = nc*x/np.sum(x,axis=1)
    y = np.zeros((n,nc,h,w),dtype=np.float32)
    for ii in xrange(n):
        for jj in xrange(nc):
            y[ii,jj,:,:] = x[ii,jj,0,0]
    return y

def trans_bmap_label01(bmap):
    # bmap: [n,1,h,w]
    # out: [n*h*w,2]
    out = np.reshape(bmap,(-1,1))
    out = np.concatenate((1-bmap,bmap),axis=-1)
    #out = np.concatenate((bmap,1-bmap),axis=1)
    #out = np.swapaxes(bmap,) 
    return np.asarray(out,dtype=np.int32)

def fun_is_detect(his_res,cur_res):
    # his_res: [n_historical_sampe+1,ntracker]
    # cur_res: [ntracker]
    mn_res = np.mean(his_res[0:-1,:],axis=0)
    factor = 0.8
    if factor*np.max(mn_res) > np.max(cur_res): 
        flag_detect = True
    else:
        flag_detect = False
    #idx = np.mod(iframe,his_res.shape[0]-1)
    return flag_detect

def fun_get_search_scale():
    s1 = [1]#,0.995,0.99,1.005,1.01]#[1,0.99,0.98,1.01,1.02,0.995,0.985,1.005,1.015]
    s2 = [1]#[1,0.99,0.995,1.01,1.005]
    n1 = len(s1)
    n2 = len(s2)
    search_scale = np.zeros((n1*n2,2),dtype=np.float32)
    count = 0
    for x in s1:
        for y in s2:
            search_scale[count,0] = x
            search_scale[count,1] = x*y 
            count = count + 1
    assert(search_scale[0,0]==1 and search_scale[0,1]==1)
    return search_scale

def fun_border_modification(ctr_rct,img_h,img_w):
    # [w,h,wlen,hlen]
    wlen = ctr_rct[2]
    hlen = ctr_rct[3]
    if ctr_rct[0] - wlen/2 < 0:
        ctr_rct[0] = wlen/2 + 0.5
    if ctr_rct[0] + wlen/2 > img_w:
        ctr_rct[0] = img_w - wlen/2 -0.5
    if ctr_rct[1] - hlen/2 < 0:
        ctr_rct[1] = hlen/2 + 0.5
    if ctr_rct[1] + hlen/2 > img_h:
        ctr_rct[1] = img_h - hlen/2 -0.5
    return ctr_rct

def fun_get_search_ctr(rct,factor=1.):
    # [w,h]
    offset_search = np.zeros((1,2),dtype = np.float32) 
    wlist = (0,) # -rct[0]*factor,rct[0]*factor)
    hlist = (0,) # -rct[1]*factor,rct[1]*factor)
    count = 0
    for w in wlist:
        for h in hlist:
            offset_search[count,:] =[w,h]
            count = count + 1
    return offset_search
    

def find_stable_tracker(response,maxres,mx_h,mx_w,loss,update_idx):
    ## weight_trackers: [ntracker]
    ## response: [ntracker,h,w] 
    ## maxres: [ntracker]
    ## maxres_hw: [ntracker,2]
    ## R: [ntracker]
    ## loss: [nsample+1,ntracker]
    
    ntracker = response.shape[0]
    nsample  = loss.shape[0]-1
    eps = 1.0e-4
        
    for ii in xrange(ntracker):
        loss[-1,ii] = maxres[ii]-response[ii,mx_h,mx_w]
    
    loss_mean = np.mean(loss[0:nsample,:],axis=0) # [ntracker]
    loss_std  = np.std(loss[0:nsample,:],axis =0) # [ntracker]
    loss_mean[loss_mean<0.0001] = 0
    loss_std [loss_std <0.0001] = 0
    
    ## compute alpha
    score_tracker = np.absolute(loss[-1,:] - loss_mean)/(loss_std+eps)

    #loss_idx = np.mod(iframe,nsample)
    loss[update_idx,:] = loss[-1,:]

    return score_tracker


def update_weights(iframe,weight_trackers,response,maxres,maxres_hw,mx_scale,mx_h,mx_w,R,loss):
    ## weight_trackers: [ntracker]
    ## response: [ntracker,h,w] 
    ## maxres: [ntracker]
    ## maxres_hw: [ntracker,2]
    ## R: [ntracker]
    ## loss: [nsample+1,ntracker]

    ntracker = response.shape[0]
    nsample  = loss.shape[0]-1
    eps = 1.0e-6
    
    for ii in xrange(ntracker):
        loss[-1,ii] = maxres[ii]-response[ii,mx_h,mx_w]

    loss_mean = np.mean(loss[0:nsample,:],axis=0) # [ntracker]
    loss_std  = np.std(loss[0:nsample,:],axis =0) # [ntracker]
    loss_mean[loss_mean<0.0001] = 0
    loss_std [loss_std <0.0001] = 0

    ## compute alpha
    curDiff = np.absolute(loss[-1,:] - loss_mean)/(loss_std+eps)

    ##
    loss_idx = np.mod(iframe,nsample)
    loss[loss_idx,:] = loss[-1,:]
    min_idx = np.argmin(curDiff)
    W = np.zeros(ntracker,dtype=np.float32)
    W[min_idx] = 1 
    return loss,W


    '''
    print curDiff
    alpha  = 0.97*np.exp(-10*curDiff)
    alpha[alpha>0.9] = 0.97
    alpha[alpha<0.12] = 0.119        

    ## compute R
    lossA = np.sum(weight_trackers*loss[-1,:]) # \bar{l}_l^k
    R = R*alpha + (1-alpha)*(lossA-loss[-1,:])

    print alpha
    print R
    ## update loss    
    loss_idx = np.mod(iframe,nsample)
    loss[loss_idx,:] = loss[-1,:]

    ## update weights
    A = 0.#0.011
    W = np.zeros(ntracker,dtype=np.float32)
    mn = np.mean(R)
    for ii in xrange(ntracker):
	x = R[ii] - mn
  	if x<=0:
	    W[ii] = 0
	else:
	    W[ii] = x/mx_scale*np.exp((x*x/(2*mx_scale)))
    W = W/np.sum(W)
    '''

    #return loss,R,W

def get_maxdirection(x):
    n = len(x)
    m = np.sum(x<0)
    r = 1.*m/n
    if r>0.5:
        y = np.mean(x[x<0])
    else:
        y = np.mean(x[x>=0])
    return y	    

def get_maxdirection2(maxres_hw):
    n = maxres_hw.shape[0]
    m = np.sum(maxres_hw<0,axis=0)
    flag = np.ones((1,2),dtype=np.float32)
    if 1.*m[0]/n>0.5:
        flag[0,0] = -1
    if 1.*m[1]/n>0.5:
        flag[0,1] = -1
    u = np.prod(maxres_hw*flag,axis=-1)
    return np.where(u>=0)[0]
       
def get_max_ps(response,ctr_h,ctr_w):
    # [h,w]
    # output: choose from [h,w]
    [h,w] = response.shape
    mxv_idx = np.argmax(response)
    maxres_hw = np.unravel_index(mxv_idx,response.shape)
    maxres    = response[maxres_hw[0],maxres_hw[1]]

    mx_h = maxres_hw[0] - ctr_h
    mx_w = maxres_hw[1] - ctr_w
    return maxres,mx_h,mx_w

def get_ps_offset(response,ctr_h,ctr_w,stable_scores):
    # [ntracker,h,w]
    # output: choose from [h,w]
    [ntracker,h,w] = response.shape
    maxres = np.zeros((ntracker),dtype=np.float32)
    maxres_hw = np.zeros((ntracker,2),dtype=np.int32)
    #mx_h = 0.
    #mx_w = 0.
    for kk in xrange(ntracker):
        mxv_idx = np.argmax(response[kk])
        maxres_hw[kk] = np.unravel_index(mxv_idx,response[kk].shape)
        maxres[kk]     = response[kk,maxres_hw[kk,0],maxres_hw[kk,1]]
        #mx_h  = mx_h + maxres_hw[kk,0]*weight_trackers[kk]
	#mx_w  = mx_w + maxres_hw[kk,1]*weight_trackers[kk]

    maxres_hw[:,0] = maxres_hw[:,0] - ctr_h
    maxres_hw[:,1] = maxres_hw[:,1] - ctr_w

    '''
    ## 1
    mx_h = get_maxdirection(maxres_hw[0:-1,0])
    mx_w = get_maxdirection(maxres_hw[0:-1,1])
    
    mx_h = 0.5*mx_h + 0.5*maxres_hw[-1,0]
    mx_w = 0.5*mx_w + 0.5*maxres_hw[-1,1]

    '''
    ## 2
    '''
    idx0 = get_maxdirection2(maxres_hw[0:-1,:])
    idx  = np.argmax(maxres[idx0])
    idx  = idx0[idx]
    mx_h = maxres_hw[idx,0]
    mx_w = maxres_hw[idx,1] 
    mx_h = 0.5*mx_h + 0.5*maxres_hw[-1,0]
    mx_w = 0.5*mx_w + 0.5*maxres_hw[-1,1]
    '''
    '''
    ## 3
    mx_h = get_maxdirection(maxres_hw[0:-1,0])
    mx_w = get_maxdirection(maxres_hw[0:-1,1])
    
    mx_h = 0.5*mx_h + 0.5*maxres_hw[-1,0]
    mx_w = 0.5*mx_w + 0.5*maxres_hw[-1,1]
    idx = 0
    '''
    ## 5
    '''
    idx  = np.argmin(stable_scores)
    mx_h = 0.5*maxres_hw[idx,0] + 0.5*maxres_hw[-1,0]
    mx_w = 0.5*maxres_hw[idx,1] + 0.5*maxres_hw[-1,1]
    '''

    ## 6/7
    '''
    idx0 = get_maxdirection2(maxres_hw[0:-1,:])
    idx  = np.argmin(stable_scores[idx0])
    idx  = idx0[idx]
    if maxres[idx]>=maxres[-1]:
    	mx_h = maxres_hw[idx,0] + 0.5*maxres_hw[-1,0]
    	mx_w = maxres_hw[idx,1] + 0.5*maxres_hw[-1,1]
    else:
        idx = -1
	mx_h = maxres_hw[-1,0]
        mx_w = maxres_hw[-1,1]
    '''
    ## 8
    idx0 = get_maxdirection2(maxres_hw[0:-1,:])
    idx  = np.argmin(stable_scores[idx0])    
    idx  = idx0[idx]
    if maxres[idx]>=maxres[-1] and stable_scores[idx]<stable_scores[-1]: # small is better
        mx_h = maxres_hw[idx,0]
        mx_w = maxres_hw[idx,1]
    else:
        idx = -1
        mx_h = maxres_hw[-1,0]
        mx_w = maxres_hw[-1,1]
    ## 9
    '''    
    idx  = np.argmax(maxres[0:-1])    
    if maxres[idx]>=maxres[-1] and stable_scores[idx]<stable_scores[-1]: # small is better
	#mx_h = maxres_hw[idx,0] + 0.5*maxres_hw[-1,0]
        #mx_w = maxres_hw[idx,1] + 0.5*maxres_hw[-1,1]
        mx_h = maxres_hw[idx,0]
        mx_w = maxres_hw[idx,1]
    else:
        idx = -1
        mx_h = maxres_hw[-1,0]
        mx_w = maxres_hw[-1,1]
    '''
    return maxres,maxres_hw,(mx_h,mx_w),idx

def get_scale(response):
    # [ntracker,nscale,h,w]
    # output: choose from [h,w]
    [ntracker,nscale,h,w] = response.shape
    maxres = np.zeros((nscale),dtype=np.float32)
    for ii in xrange(nscale):
        maxres[ii] = np.max(response[-1,ii,:,:])
	#for kk in xrange(ntracker): 
	#    maxres[ii] = maxres[ii] + weight_trackers[kk]* np.max(response[kk,ii,:,:])
    return np.argmax(maxres)
	 
def get_max_scale(response):
    # [ntracker,nscale,h,w]
    # output: choose from [h,w]
    [nscale,h,w] = response.shape
    maxres = np.zeros((nscale),dtype=np.float32)
    #maxres = response[:,h//2,w//2]
    hc,wc = h//2,w//2
    for ii in range(nscale):
    #    maxres[ii] = np.mean(response[ii,hc-1:hc+2,wc-1:wc+2]) 
    #    maxres[ii] = np.max(response[ii,:,:])
        mxv_idx = np.argmax(response[ii,:,:])
        cc = np.unravel_index(mxv_idx,[h,w])
        maxres[ii]    = np.mean(response[ii,cc[0]-1:cc[0]+2,cc[1]-1:cc[1]+2])
    return np.argmax(maxres),maxres

def get_max_offset(responses):
    ## [noffset,ntracker,nscale,h,w]
    ## output: choose from [n_sfh,n_sfw]
    [noffset,nscale,h,w] = responses.shape
    mxv0 = np.zeros((noffset),dtype=np.float32)
    for ioffset in range(noffset):
        mxv0[ioffset] = mxv0[ioffset] + np.max(responses[ioffset])

    #print mxv0

    return np.argmax(mxv0)

def get_ctr_shift_trackers(responses):
    ## [noffset,ntracker,nscale,h,w]
    ## output: choose from [n_sfh,n_sfw]
    [noffset,ntracker,nscale,h,w] = responses.shape
    mxv0 = np.zeros((noffset),dtype=np.float32)
    for ioffset in range(noffset):
        for itracker in range(ntracker): # current tracker is considered
    	    mxv0[ioffset] = mxv0[ioffset] + np.max(responses[ioffset,itracker])

    #print mxv0
    
    return np.argmax(mxv0) # divide/ntracker


def close_logger(logger):
    for xx in logger.handlers[:]:
        xx.close()
        logger.removeHandler(xx)

def fun_get_strnn_list(srnn_directions):
    assert(srnn_directions[-1]=='current')
    n = len(srnn_directions)
    trnn_flag = False
    if n>0:
        strnn_list = []
        for x in srnn_directions:
            assert(x=='topleft' or x=='bottomright' or x=='topright' or x=='bottomleft' or x=='previous' or x=='current')
            strnn_list.append((x,'current'))
            if x=='previous':
                trnn_flag = True
    else:
        strnn_list = [('current',)]
    return strnn_list,trnn_flag    

def fun_get_patch(x,ctr,sz):
    # [w,h]
    h = x.shape[0]
    w = x.shape[1]
    hidx = np.int32(np.floor(ctr[1]-sz[1]/2+0.5) + np.arange(sz[1]))
    widx = np.int32(np.floor(ctr[0]-sz[0]/2+0.5) + np.arange(sz[0]))
    hidx[hidx<0] = 0
    hidx[hidx>h-1] = h-1
    widx[widx<0] = 0
    widx[widx>w-1] = w-1
    #print hidx, widx
    [hidx,widx] = np.meshgrid(hidx,widx)
    hidx = hidx.transpose()
    widx = widx.transpose()
    #print hidx, widx
    return np.asarray(x[hidx,widx,:],dtype=np.float32)  

def fun_get_patch_2pts(x,wh_1, wh_2):
    h = x.shape[0]
    w = x.shape[1]
    hidx = np.int32(np.arange(wh_1[1],wh_2[1]))
    widx = np.int32(np.arange(wh_1[0],wh_2[0]))
    hidx[hidx<0] = 0
    hidx[hidx>h-1] = h-1
    widx[widx<0] = 0
    widx[widx>w-1] = w-1
    #print hidx, widx
    [hidx,widx] = np.meshgrid(hidx,widx)
    hidx = hidx.transpose()
    widx = widx.transpose()
    #print hidx, widx
    return np.asarray(x[hidx,widx,:],dtype=np.float32)

def fun_response(model_w,xf,zf,kernel_type,sigma,gamma):
    # xf: nx*(nd*c)*h*w, frequency domain
    # zf: nz*(nd*c)*h*w, frequency domain
    # model_w: (2*nx)*1*h*w, real domain
    # output: nz*h*w, real_domain
    h = xf.shape[-2]
    w = xf.shape[-1]
    nz = zf.shape[0]
    nx = xf.shape[0]
    if nx==1:
        assert(nz==1)
        assert(model_w.shape[0]==h and model_w.shape[1]==w)
        k_zf_xf = fun_correlation_singlesample(zf[0],xf[0],kernel_type,sigma) # h*w
        return np.real(np.fft.ifft2(k_zf_xf*model_w))
    else:
        kernel_zf_xf = fun_kernel_multisample(xf,zf,kernel_type,sigma) # (2*nz)*(2*nx)*h*w
        response = np.zeros((2*nz,h,w))
        for ii in xrange(h):
            for jj in xrange(w):
                response[:,ii,jj] = (np.dot(kernel_zf_xf[:,:,ii,jj],model_w[:,:,ii,jj])).flatten() 
        return np.real(np.fft.ifft2(response[0:nz,:,:]+1.0j*response[nz::,:,:],axes=(-2,-1)))

def fun_w(xf,yf,kernel_type,sigma,gamma):
    # xf: n*(nd*c)*h*w, frequency domain
    # yf: n*1*h*w frequency domain
    # output: w >> (2*n)*1*h*w, =(I-(gamma*I+A'A)^-1*A'A)*y
    assert(xf.shape[2:]==yf.shape[2:] and xf.shape[0]==yf.shape[0])
    n = xf.shape[0]
    h = xf.shape[-2]
    w = xf.shape[-1]
    if n==1:
        k_xf = fun_correlation_singlesample(xf[0],xf[0],kernel_type,sigma) + gamma # h*w
        pjt = np.divide(yf[0,0],k_xf) 	
    else:
        AtA = fun_kernel_multisample(xf,xf,kernel_type,sigma)
        pjt = np.zeros((2*n,1,h,w),dtype=np.float32)
        gammaI = gamma*np.eye(2*n,dtype = np.float32)
    
        y = np.concatenate((np.real(yf),np.imag(yf)),axis=0) 
        for ii in xrange(h):
            for jj in xrange(w):
                pjt[:,:,ii,jj] = y[:,:,ii,jj] - np.dot(np.dot(np.linalg.inv(AtA[:,:,ii,jj] + gammaI),AtA[:,:,ii,jj]),y[:,:,ii,jj])
    return pjt

def fun_kernel_multisample(xf,yf,kernel_type,sigma):
    # xf: nx*(nd*c)*h*w, frequency domain
    # yf: ny*(nd*c)*h*w or 1*(nd*c)*h*w, frequency domain
    # out_kernel: (2*ny)*(2*nx)*h*w, for each position yf'*xf
    assert(xf.shape[1:]==yf.shape[1:])
    h = xf.shape[-2]
    w = xf.shape[-1]
    nx = xf.shape[0]
    ny = yf.shape[0]
    out_kernel = np.zeros((ny*2,nx*2,h,w),dtype=np.float32)
 
    if kernel_type == 'gaussian':
        for ii in xrange(h):
            for jj in xrange(w):
                xx = xf[:,:,ii,jj]
                yy = yf[:,:,ii,jj]
                d_yr_xr = cdist(np.real(yy),np.real(xx),'sqeuclidean')
                d_yr_xi = cdist(np.real(yy),np.imag(xx),'sqeuclidean')
                d_yi_xr = cdist(np.imag(yy),np.real(xx),'sqeuclidean')
                d_yi_xi = cdist(np.imag(yy),np.imag(xx),'sqeuclidean')

                out_kernel[0:ny,0:nx,ii,jj] = d_yr_xr + d_yi_xi
                out_kernel[0:ny,nx::,ii,jj] = d_yr_xi - d_yi_xr
                out_kernel[ny::,0:nx,ii,jj] = d_yi_xr - d_yr_xi
                out_kernel[ny::,nx::,ii,jj] = d_yr_xr + d_yi_xi
        out_kernel = np.exp(-out_kernel/(sigma*sigma*h*w)) # ??
    elif kernel_type == 'linear':
        for ii in xrange(h):
            for jj in xrange(w):
                xx = xf[:,:,ii,jj]
                yy = yf[:,:,ii,jj]
                d_yr_xr = np.dot(np.real(yy),(np.real(xx)).transpose())
                d_yr_xi = np.dot(np.real(yy),(np.imag(xx)).transpose())
                d_yi_xr = np.dot(np.imag(yy),(np.real(xx)).transpose())
                d_yi_xi = np.dot(np.imag(yy),(np.imag(xx)).transpose())

                out_kernel[0:ny,0:nx,ii,jj] = d_yr_xr + d_yi_xi
                out_kernel[0:ny,nx::,ii,jj] = d_yr_xi - d_yi_xr
                out_kernel[ny::,0:nx,ii,jj] = d_yi_xr - d_yr_xi
                out_kernel[ny::,nx::,ii,jj] = d_yr_xr + d_yi_xi
        out_kernel = out_kernel/(h*w) # ???
    else:
        assert(kernel_type=='gaussian' or kernel_type == 'linear')
    return out_kernel


def fun_correlation_singlesample(xf,yf,kernel_type,sigma):
    # xf:   c*h*w, frequency domain
    # yf:   c*h*w, frequency domain
    # sigma: gaussian param
    # kernel_type: 'linear', or 'gaussian' 
    # output: h*w
    # Note: yf.conjugate()
    assert(xf.shape==yf.shape)
    N = xf.shape[1]*xf.shape[2]
    M = N*xf.shape[0]
    if kernel_type == 'linear':
        kf = np.sum(xf*yf.conjugate(),axis=0)/M
    elif kernel_type == 'gaussian':

        N = xf.shape[-2]*xf.shape[-1]
        xx = np.real((xf.conjugate()*xf).sum())/N # n*nd*c*1*1
        yf_conj = yf.conjugate()
        yy = np.real((yf*yf_conj).sum())/N

        xyf = xf*yf_conj
        xy = np.sum(np.real(np.fft.ifft2(xyf,axes=(-2,-1))),axis=0)

        uu = (xx+yy-2*xy)/M
        uu = uu*(uu>0)
        kf = np.fft.fft2(np.exp( (-1/(sigma*sigma))*uu ),axes=(-2,-1))
    else:
        assert(kernel_type=='gaussian' or kernel_type=='linear')
    return kf

def fun_precision_location(pre_ctr_rcts,gt_ctr_rcts,thre=50,step=1):
    ## pre_ctr_rcts, gt_ctr_rcts: ctr 
    n = pre_ctr_rcts.shape[0]
    #### ctr localization
    diff_ps = np.sqrt(np.sum(np.square(pre_ctr_rcts[:,0:2]-gt_ctr_rcts[:,0:2]),axis=-1))
    x = np.arange(0,thre+step/2,thre)
    m = len(x)
    ratio_ps = np.zeros((m,2),dtype=np.float32) 
    ratio_ps[:,0] = x
    for ii in range(m):
        pc = x[ii]
        ratio_ps[ii,1] = np.sum(diff_ps<=pc) 
    return np.mean(diff_ps),ratio_ps,diff_ps

def fun_precision_overlap(pre_ctr_rcts,gt_ctr_rcts,step=0.05):
    #### overlap
    assert(pre_ctr_rcts.shape==gt_ctr_rcts.shape) 
    n = pre_ctr_rcts.shape[0]
    ## from ctr to box
    pbbx = fun_ctr2rct(pre_ctr_rcts)
    pbbx[:,2] = pbbx[:,0] + pbbx[:,2]
    pbbx[:,3] = pbbx[:,1] + pbbx[:,3]
    gbbx  = fun_ctr2rct(gt_ctr_rcts)
    gbbx[:,2]  = gbbx[:,0] + gbbx[:,2]
    gbbx[:,3]  = gbbx[:,1] + gbbx[:,3]

    lt = np.dstack((pbbx[:,0:2],gbbx[:,0:2]))
    br = np.dstack((pbbx[:,2:4],gbbx[:,2:4]))
    ## 
    whs_min = np.min(lt,axis=-1)
    whs_max = np.max(lt,axis=-1) 
    whe_min = np.min(br,axis=-1) 
    whe_max = np.max(br,axis=-1)  

    ## intersection 
    inner = whe_min-whs_max
    inner[inner<0] = 0
    inner = np.prod(inner,axis=-1) 

    ## union
    outer = whe_max - whs_min
    outer[outer<0] = 0
    outer = np.prod(outer,axis=-1)
    outer[outer<=0] = 1

    diff = np.divide(inner,outer)
    x = np.arange(0,1+step/2,step)
    m = len(x)
    ratio = np.zeros((m,2),dtype=np.float32)
    ratio[:,0] = x
    for ii in range(m):
        k = np.sum(diff>x[ii])
        ratio[ii,1] = 1.*k/n
    return np.mean(diff),ratio,diff
    
def fun_shift_feas(x,px,directions,step,outx):
    ## x,px: n*c*(h+2*step)*(w+2*step)
    ## outx: n*(ndirections*c)*h*w
    xs = x.shape
    os = outx.shape
    assert(xs[0]==os[0] and xs[2]==os[2]+2*step and xs[3]==os[3]+2*step)
    assert(xs[1]*len(directions)==os[1])
    c = xs[1]
    h = os[2]
    w = os[3]
    kk = 0
    for dct in directions:	
        if dct == 'topleft':
            outx[:,kk:kk+c,:,:] = x[:,:,0:h,0:w]
        elif dct == 'topright':
            outx[:,kk:kk+c,:,:] = x[:,:,0:h,2*step::]
        elif dct == 'bottomleft':
            outx[:,kk:kk+c,:,:] = x[:,:,2*step::,0:w]
        elif dct == 'bottomright':
            outx[:,kk:kk+c,:,:] = x[:,:,2*step::,2*step::]
        elif dct == 'current':
            outx[:,kk:kk+c,:,:] = x[:,:,step:-step,step:-step]
        elif dct == 'previous':
            outx[:,kk:kk+c,:,:] = px[:,:,step:-step,step:-step]
        else:
            assert(dct=='topleft' or dct =='topright' or dct =='bottomleft' or dct == 'bottomright' or dct == 'current' or dct == 'previous')
        kk = kk+c
    return

def fun_get_freq_fea(x,is_cos_window):
# x: n*directions*c*h*w
    h = x.shape[-2]
    w = x.shape[-1]
    if is_cos_window:
        shp = np.ones(len(x.shape))	
        shp[-2] = h
        shp[-1] = w
        x = x*np.reshape(fun_cos_win(h,w),shp)
    xf = np.fft.fft2(x,axes=(-2,-1))
    return xf

def fun_cos_win(h,w,shp):
    x = np.arange(h)
    x = x.reshape((h,1))
    x = 0.5*(1-np.cos((2*np.pi/(h-1))*x))
    y = np.arange(w)
    y = y.reshape((1,w))
    y = 0.5*(1-np.cos((2*np.pi/(w-1))*y))
    xy = np.dot(x,y)
    xy = np.asarray(xy,dtype=np.float32)
    #save_mat_file('1.mat',xy,None,None,None)
    #assert(3==1)
    #xy = xy.reshape((h,w,1))
    return np.reshape(xy,shp)

def fun_draw_polygon_on_image(im,fname, xys1=None, xys2=None):
    if len(im.shape)==2:
        im = np.dstack((im,im,im))
    im = Image.fromarray(im)
    dr = ImageDraw.Draw(im)
    if xys1 is not None:
        dr.polygon( xys1, outline = "blue")
    if xys2 is not None:
        dr.polygon( xys2, outline = "red")
    del dr
    im.save(fname)


def fun_draw_rct_on_image(im,fname,rct1,rct2,rct3):
    if len(im.shape)==2:
        im = np.dstack((im,im,im))
    im = Image.fromarray(im)
    dr = ImageDraw.Draw(im)
    if rct1 is not None:
        ws = rct1[0]-np.floor(rct1[2]/2)
        hs = rct1[1]-np.floor(rct1[3]/2)
        dr.rectangle(((ws,hs),(ws+rct1[2],hs+rct1[3])), outline = "blue")
    if rct2 is not None:
        ws = rct2[0]-np.floor(rct2[2]/2)
        hs = rct2[1]-np.floor(rct2[3]/2)
        dr.rectangle(((ws,hs),(ws+rct2[2],hs+rct2[3])), outline = "green")
    if rct3 is not None:
        ws = rct3[0]-np.floor(rct3[2]/2)
        hs = rct3[1]-np.floor(rct3[3]/2)
        dr.rectangle(((ws,hs),(ws+rct3[2],hs+rct3[3])), outline = "red")
    del dr
    im.save(fname)

def fun_draw_mask_on_image(im,fname,bmap,rct_ctr,win_sz):
    h = im.shape[0]
    w = im.shape[1]
    if len(im.shape)==2:
        im = np.dstack((im,im,im))

    sz = np.int32(win_sz)
    hs = np.max((0,rct_ctr[1]-sz[1]/2+0.5))
    he = np.min((h-1,hs + sz[1]))

    ws = np.max((0,rct_ctr[0] - sz[0]/2+0.5))
    we = np.min((w-1,ws + sz[0]))
    
    hs = np.uint32(hs)
    he = np.uint32(he)
    ws = np.uint32(ws)
    we = np.uint32(we)
    #sz = np.uint32(win_sz)
    x = MISC.imresize(np.uint8(bmap)*255,np.uint32([he-hs,we-ws]),interp='bicubic')
    y = im[hs:he,ws:we,:]
    hidx, widx = np.where(x>128)
    #print len(hidx),len(widx),hs,he,ws,we
    y[hidx,widx,:] = 255
    
    im[hs:he,ws:we,:] = y

    im = Image.fromarray(im)
    im.save(fname)


def fun_get_max_response(x1,x2,x3,x4):
    max_idx = np.argmax(x1)
    [c1,h1,w1] = np.unravel_index(max_idx,x1.shape)
    max_v1 = x1[c1,h1,w1]

    max_idx = np.argmax(x2)
    [c2,h2,w2] = np.unravel_index(max_idx,x2.shape)
    max_v2 = x2[c2,h2,w2]

    max_idx = np.argmax(x3)
    [c3,h3,w3] = np.unravel_index(max_idx,x3.shape)
    max_v3 = x3[c3,h3,w3]

    max_idx = np.argmax(x4)
    [c4,h4,w4] = np.unravel_index(max_idx,x4.shape)
    max_v4 = x4[c4,h4,w4]
    
    v = (max_v1,max_v2,max_v3,max_v4)
    max_idx = np.argmax(v)
    max_v = v[max_idx]
    idx = ((c1,h1,w1),(c2,h2,w2),(c3,h3,w3),(c4,h4,w4))
    return max_v, idx[max_idx]

def fun_binarize_map(prop):
    prop[prop>=0.5] = 1
    prop[prop<=0.5] = 0
    return prop 

def fun_get_binary_map(win_size,rct_size,cell_size):
    #win_ctr = np.floor(win_size/2)

    win_size2 = np.floor(win_size/cell_size+0.5)
    rct_size2 = np.floor(rct_size/cell_size)
    win_ctr2  = np.int32(np.floor(win_size2/2+0.5))-1

    h = win_size2[1]
    w = win_size2[0]
    ctrh = win_ctr2[1]
    ctrw = win_ctr2[0]

    m = np.zeros((h,w),dtype=np.float32)

    hs = ctrh - np.floor(rct_size2[1]/2)
    he = hs + rct_size2[1]
    ws = ctrw - np.floor(rct_size2[0]/2)
    we = ws + rct_size2[0]

    m[hs:he,ws:we] = 1.0
    return m

def fun_scale_model_te(x, w, m_type):
    ## x: nscale*nexpert*d
    ## w: np*d*1 or np*ns*d

    ns, npp, d = x.shape
    if m_type == 'expert':
        pred = np.zeros((npp,ns),dtype=np.float32)
        for ii in range(npp):
            p = np.dot(x[:,ii,:],w[ii,:,:]) # ns*1
            pred[ii,:] = p.flatten()
        return pred
    elif m_type == 'dot':
        pred = np.zeros((npp,ns),dtype=np.float32)
        for ii in range(npp):
            p = np.sum(x[:,ii,:]*w[ii,:,:],axis=1) # ns*d --> ns*1
            pred[ii,:] = p
        return pred
    else:
        assert(m_type=='dot' and m_type == 'expert')

def fun_scale_model_tr(x, y, gamma, m_type):
    ## x: nscale*nexpert*d
    ## y: nscale*1
    ## w: 
    ns, npp, d = x.shape
 
    gI = np.eye(d, dtype = np.float32) * np.float32(gamma)

    if m_type == 'expert':
        ws = []
        for ii in range(npp):
            xi = x[:,ii,:] # nscale*d
            xtx = np.dot(np.transpose(xi),xi) + gI
            invx = inv(xtx)
            xty = np.dot(invx,np.dot(np.transpose(xi), y)) # d*1
            ws.append(xty)
        return np.stack(ws,axis=0) # np*d*1
    elif m_type == 'dot':
        ws = []
        for ii in range(npp):
            xi = x[:,ii,:] # nscale*d
            w = xi*y/(np.sum(xi*xi,axis=1, keepdims=True)+gamma)
            ws.append(w)
        return np.stack(ws,axis=0) # np*ns*d
    else:
        assert(m_type=='dot' and m_type == 'expert')

  
def fun_get_dynamic_scale(win_sz):
    '''
    offset = np.asarray([-7,-5,-4,-3,-2,-1,0,1, 2, 3,4,5,7])
    n = len(offset)
    k = np.argmax(win_sz)
    win_szs = np.zeros((n*3,2))
    win_sls = np.zeros((n*3,2))
    count = 0
    for ii in np.arange(n):
        offset1 = offset[ii]
        jj_start, jj_end = np.max([0,ii-1]), np.min([ii+2,n])
        for jj in np.arange(jj_start,jj_end):
            offset2 = offset[jj]
            if k==0:
               sz_offset = np.asarray([offset1,offset2])
            elif k==1:
               sz_offset = np.asarray([offset2,offset1])
            else:
               assert('error')
            win_szs[count,:] = win_sz + sz_offset
            win_sls[count,:] = win_szs[count,:]/win_sz
            count = count+1 

    '''
    ss = [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8]
    n = len(ss)
    win_szs = np.zeros((n,2))
    win_sls = np.zeros((n,2))
    v = np.max(win_sz)
    count = 0
    for k in ss:
        r = 1 + 1.0*k/v
        win_szs[count,:] = win_sz * r
        win_sls[count,:] = r
        count = count + 1
    return win_szs[0:count,:], win_sls[0:count,:], count   

def fun_get_scale_map(target_sz,nScales,scale_sigma_factor,scale_step):
    # ys: nScales*1
    scale_sigma = np.sqrt(nScales)*scale_sigma_factor
    ss = np.arange(1,nScales+1) - np.ceil(nScales/2)
    #pdb.set_trace()
    ys = np.exp(-0.5*(ss*ss)/(scale_sigma*scale_sigma))
    idx = np.argmax(ys)
    assert idx == np.int32(nScales/2)

    ## scale cos window
    assert nScales%2 == 1
    scale_window = np.array(np.hanning(nScales),np.float32)

    ## scale factors
    ss = np.arange(1,nScales+1)
    x = np.int32(nScales/2)
    scaleFactors = scale_step**(np.ceil(nScales/2)-ss)
    
    ## resize dim
    #scale_model_factor = 1;
    #if np.prod(target_sz) > scale_model_max_area:
    #    scale_model_factor = np.sqrt(scale_model_max_area/np.prod(target_sz))
    #scale_model_sz = np.floor(target_sz * scale_model_factor)
    #scale_model_sz = fea_sz

    return np.reshape(ys,(-1,1)),np.reshape(scale_window,(-1,1)),scaleFactors

def find_weight_idx(nn_map_idx,vgg_out_layers,vgg_out_layers2,K):
    n = len(vgg_out_layers)
    n2 = len(vgg_out_layers2)
    idx = []
    ref = np.asarray(vgg_out_layers)
    for ii in range(n2):
        kk = np.where(ref==vgg_out_layers2[ii])[0]
        if len(kk) == 0:
            assert('vgg_out_layers2 not exist in vgg_out_layer')
        ss, ee = K*nn_map_idx[kk], K*nn_map_idx[kk+1]
        idx.append(np.arange(ss,ee))
    return np.concatenate(idx)
    

def fun_compute_min_max_scale_factor(win_sz, target_sz, im_sz, scale_step):
    # w,h
    min_scale_factor = scale_step **np.ceil(np.log(np.max(20./win_sz)) / np.log(scale_step));
    max_scale_factor = scale_step **np.floor(np.log(np.min(1*im_sz/ target_sz)) /np.log(scale_step))

    #min_scale_factor = -(1-min_scale_factor)*np.min(target_sz)/2
    #max_scale_factor = (max_scale_factor-1)*np.max(target_sz)/2

    return min_scale_factor, max_scale_factor

def fun_get_peak_map(win_size,rct_size,cell_size,fea_sz,isroll):
# input >> [width,height]
# output >> matrix of [h,w]

    sigma_factor = 0.1
    sigma = np.sqrt(rct_size[1]*rct_size[0])*sigma_factor/cell_size

    #win_size2 = np.floor(win_size/cell_size+0.5)
    win_size2=fea_sz 
    [rs,cs] = np.meshgrid( np.arange(1,win_size2[0]+1)-np.floor(win_size2[0]/2.+0.5),\
                           np.arange(1,win_size2[1]+1)-np.floor(win_size2[1]/2.+0.5) )
    m = np.exp(-0.5/(sigma*sigma)*(rs*rs+cs*cs))
    m = m.astype(np.float32)

    if isroll:
        m = np.roll(m, -np.int32(np.floor(win_size2[1]/2.+0.5))+1,axis=0)
        m = np.roll(m, -np.int32(np.floor(win_size2[0]/2.+0.5))+1,axis=1)
        ctr_h = 0
        ctr_w = 0
        assert m[0,0]==1
    else:
        ctr_h = np.int32(np.floor(win_size2[1]/2.+0.5))-1
        ctr_w = np.int32(np.floor(win_size2[0]/2.+0.5))-1
        assert m[ctr_h,ctr_w] == 1
    return m,ctr_h,ctr_w

def fun_get_search_window2(target_sz,im_sz,magh=None,magw=None):
    ## [width, height]
    ## if padding_type==None: use all other three vars
    ## else use target_sz, padding
    
    if magh is None or magw is None:
        if target_sz[1]/im_sz[1] > 0.3:
            magh = 1.4
            magw = 2
        elif np.max(target_sz) < 30:
             magh = 3
             magw = 3
        elif target_sz[1]/target_sz[0]>2:
            magh = 2#1.4
            magw = 2
        else:
            magh = 2.4
            magw = 2.4
    '''
    if magh is None or magw is None:
        if target_sz[1]/target_sz[0] > 2:
            magh = 1.4
            magw = 3.2
        elif np.prod(target_sz)/np.prod(im_sz) >0.05:
             magh = 2
             magw = 2
        else:
             magh=3.2#3.2
             magw=3.2#3.2
        #print target_sz, magw, magh
    '''
    sz = target_sz*np.array([magw,magh])
    mxsz = np.max(sz)
    magh = mxsz/target_sz[1]
    magw = mxsz/target_sz[0]

    window_sz = np.floor(target_sz*np.array([magw,magh])+0.5) ## ??? cuizhen
    return window_sz, magh, magw



def fun_resize_samples(X,rcts,padding):
    if np.sqrt(gt_rcts[0,2]*gt_rcts[0,3])>80:
        n = X.shape[0]
        oh = X.shape[1]
        ow = X.shape[2]
        if len(X.shape)==4:
            isColor = True
        else:
            isColor = False

        sc = 80.0/np.amax(gt_rcts[0,2:])
        scale_factor = np.asarray([sc,sc],dtype=np.float32)
        h = np.int16(np.floor(oh*scale_factor[1]))# +0.5
        w = np.int16(np.floor(ow*scale_factor[0]))# +0.5
        if isColor == True:
            Z = np.zeros((n,h,w,3),dtype=X.dtype)
            for ii in range(n):
                Z[ii,:,:,:] = MISC.imresize(X[ii,:,:,:],[h,w],interp='bicubic')
        else:
            Z = np.zeros((n,h,w),dtype=X.dtype)
            for ii in range(n):
                Z[ii,:,:] = MISC.imresize(X[ii,:,:],[h,w],interp='bicubic')

        rcts = np.zeros((n,4),dtype=np.float32)
        rcts[:,0] = gt_rcts[:,0]*scale_factor[0]
        rcts[:,1] = gt_rcts[:,1]*scale_factor[1]
        rcts[:,2] = gt_rcts[:,2]*scale_factor[0]
        rcts[:,3] = gt_rcts[:,3]*scale_factor[1]
        rcts = np.floor(rcts) ## floor??

    else:
        rcts = np.copy(gt_rcts)
        Z = np.copy(X)
        scale_factor = np.asarray([1,1],dtype=np.float32)

    nm_rct_size = np.copy(rcts[0,2:])
    nm_win_size = np.floor(nm_rct_size*(1+padding))

    return Z,scale_factor,nm_rct_size,nm_win_size,rcts

