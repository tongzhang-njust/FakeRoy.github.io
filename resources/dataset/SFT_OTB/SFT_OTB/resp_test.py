import numpy as np
import  scipy.io as sio
import math
import pdb
import scipy.io as sio

def get_max_ps(response):
    # [sl, al h, w]
    # output: choose from [h,w]
    #[h,w] = response.shape
    sl, al, h, w = response.shape
    maxres_h = np.zeros((sl,al,1,1), dtype=np.int32)
    maxres_w = np.zeros((sl,al,1,1), dtype=np.int32)
    maxres    = np.zeros((sl,al,1,1), dtype = np.float32)
    for ii in range(sl):
        for jj in range(al):
           mxv_idx = np.argmax(response[ii,jj])
           maxres_h[ii,jj,:,:],maxres_w[ii,jj,:,:] = np.unravel_index(mxv_idx, [h,w])
           maxres[ii,jj,:,:]    = response[ii, jj,maxres_h[ii,jj,:,:],maxres_w[ii,jj,:,:]]
    return maxres_h, maxres_w, maxres

def resp_newton(response, use_sz, iterations = 5):
    ## response: [sl, al, h, w], not circleshift 
    ## use_sz: [h, w]
    ##  
    pi=np.pi
    sl, al = response.shape[0],response.shape[1]
    h,w = use_sz[0], use_sz[1]
    assert(np.mod(h,2)==1 and np.mod(w,2)==1)

    ##
    #response = response.transpose((2,3,0,1)) # [sl,al,h,w]
    ## shift
    response = np.roll(response, -np.int32(np.floor(h/2.+0.5))+1,axis=-2)
    response = np.roll(response, -np.int32(np.floor(w/2.+0.5))+1,axis=-1)
    responsef = np.fft.fft2(response, axes = (-2,-1))/(h*w) # [sl,al, h, w]

    ky = np.arange(h) - np.floor(h/2.)
    ky = np.roll(ky, -np.int32(np.floor(h/2.+0.5))+1)
    ky = np.expand_dims(ky,0)  # [1, h]
    kx = np.arange(w) - np.floor(w/2.)
    kx = np.roll(kx, -np.int32(np.floor(w/2.+0.5))+1)
    kx = np.expand_dims(kx,1)  # [w, 1]

    ## max value
    row, col, init_max_response = get_max_ps(response) # [sl,al,1,1]
    #mxv_idx   = np.argmax(response, axes = (-2,-1))
    #maxres_hw = np.unravel_index(mxv_idx,response.shape) # 
    #maxres_hw = np.stack(maxres_hw) # 
    #row, col = maxres_hw[:,-2], maxres_hw[:,-1]
    #row = np.reshape(row, [sl, al, 1])  # [sl, al, 1]
    #col = np.reshape(col, [sl, al, 1])  # [sl, al, 1]

    ##
    trans_row = np.mod(row + np.floor((use_sz[0] - 1)/2), use_sz[0]) - np.floor((use_sz[0]-1)/2)
    trans_col = np.mod(col + np.floor((use_sz[1] - 1)/2), use_sz[1]) - np.floor((use_sz[1]-1)/2)

     
    init_pos_y = 2 * pi * trans_row / use_sz[0] # [sl,al,1,1]
    init_pos_x = 2 * pi * trans_col / use_sz[1] # [sl,al,1,1]
     
     

    max_pos_y = init_pos_y
    max_pos_x = init_pos_x

    ky = np.reshape(ky, (1,1,1,h))
    kx = np.reshape(kx, (1,1,w,1))
    exp_iky = np.exp(1j * ky * max_pos_y) # [1,1,1,h]*[sl,al,1,1] = [sl, al, 1, h]
    exp_ikx = np.exp(1j * kx * max_pos_x) # [1,1,w,1]*[sl,al,1,1] = [sl, al, w, 1]
     
    ky2 = ky * ky # [1,1,1,h]
    kx2 = kx * kx # [1,1,w,1]

    iter=1
    for i in range(iterations):
        # Compute gradient
        ky_exp_ky = ky * exp_iky # [sl, al, 1, h ]
        kx_exp_kx = kx * exp_ikx # [sl, al, w, 1 ]
        y_resp = np.matmul(exp_iky, responsef) # [sl,al,1,h] * [sl, al,h,w] = [sl, al, 1, w]
        resp_x = np.matmul(responsef, exp_ikx) # [sl,al,h,w] * [sl, al,w,1] = [sl, al, h, 1]
        grad_y = -np.imag(np.matmul(ky_exp_ky, resp_x)) # [sl, al, 1, h] * [sl, al, h, 1] = [sl, al, 1, 1]
        grad_x = -np.imag(np.matmul(y_resp, kx_exp_kx)) # [sl, al, 1, w] * [sl, al, w, 1] = [sl, al, 1, 1]

        ## Hessian
        ival = 1j * (np.matmul(exp_iky, resp_x)) # [sl, al, 1, h] * [sl, al, h, 1] = [sl, al, 1, 1]

        H_yy = np.real(-np.matmul(ky2 * exp_iky,resp_x) + ival) # [1,1,1,h]*[sl,al,1,h], [sl,al,h,1] = [sl, al, 1,1]
        H_xx = np.real(-np.matmul(y_resp, kx2  * exp_ikx) +ival) # [sl,al,1,w], [1,1,w,1]*[sl,al,w,1] = [sl,al,1,1]

        H_xy = np.real(-np.matmul(ky_exp_ky, np.matmul(responsef, kx_exp_kx))) # [sl,al,1,h], [sl,al,h,w], [sl,al,w,1] = [sl,al,1,1]
        det_H = H_yy * H_xx - H_xy * H_xy # [sl,al,1,1]
    
        # Compute new position using newtons method
        max_pos_y = max_pos_y - (H_xx * grad_y - H_xy * grad_x) / det_H # [sl, al, 1, 1]
        max_pos_x = max_pos_x - (H_yy * grad_x - H_xy * grad_y) / det_H # [sl, al, 1, 1]
    
        #% Evaluate maximum
        exp_iky = np.exp(1j * ky * max_pos_y) # [1,1,1,h] * [sl, al, 1, 1] = [sl, al, 1, h]
        exp_ikx = np.exp(1j * kx * max_pos_x) # [1,1,w,1] * [sl, al, 1, 1] = [sl, al, w, 1]
  
        iter = iter + 1

    max_response = np.real(np.matmul(np.matmul(exp_iky, responsef),exp_ikx)) # [sl,al,1,h],[sl,al,h,w],[sl,al,w,1] = [sl,al,1,1]

    #check for scales that have not increased in score
    ind = max_response < init_max_response
    max_response[ind] = init_max_response[ind]
    max_pos_y[ind] = init_pos_y[ind]
    max_pos_x[ind] = init_pos_x[ind]
    
    srind = np.argmax(max_response[:,:,0,0])
    sind, rind = np.unravel_index(srind,(sl,al))

    disp_row = (np.mod(max_pos_y[sind,rind,0, 0] + pi, 2*pi) - pi) / (2*pi) * use_sz[0]
    disp_col = (np.mod(max_pos_x[sind,rind,0, 0] + pi, 2*pi) - pi) / (2*pi) * use_sz[1]

    return sind, rind, disp_row, disp_col, max_response[:,:,0,0]


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    python:matlab_style_gauss2D((5,5),1)==matlab:fspecial('gaussian',5,1)
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
"""
def main():
    use_sz = [39,39]
    s = 5
    a = 1 
    
    #load_data=sio.loadmat('rand.mat')
    #randshift=np.squeeze(load_data['randshift'])
    #randidx=np.squeeze(load_data['randidx'])
    #randomn=np.squeeze(load_data['randomn'])
    
    randshift=np.random.randint(-5,5,5)
    randidx=np.random.randint(0,2,5)
    randomn=np.random.randn(5)
    sio.savemat('rand.mat',{'randshift':randshift,'randidx':randidx,'randomn':randomn})

    response = np.zeros((use_sz[0],use_sz[1],s),dtype='float')
    for ii in range(s):
        xx = matlab_style_gauss2D((use_sz[0],use_sz[1]),1)
        xx = np.roll(xx,randshift[ii] ,axis=randidx[ii])+randomn[ii]
        #xx = np.roll(xx,np.random.randint(-5,5) ,axis=np.random.randint(0,2))
        #xx += np.random.randn(1)
        response[:,:,ii]=xx
    pdb.set_trace()
    response=np.expand_dims(response,3)
    [disp_row, disp_col, scale_ind,angle_ind]=resp_newton(response, use_sz)
    print ('disp_row:%f,disp_col:%f,scale_ind:%i,angle_ind:%i')%(disp_row,disp_col,scale_ind,angle_ind)
"""
def main():
    use_sz = [39,39]
    s = 7
    a = 5 
    
    randshift=np.random.randint(-5,5,(7,5))
    randidx=np.random.randint(0,2,(7,5))
    randomn=np.random.randn(7,5)
    sio.savemat('rand.mat',{'randshift':randshift,'randidx':randidx,'randomn':randomn})

    response = np.zeros((use_sz[0],use_sz[1],s,a),dtype='float')
    for ii in range(s):
      for jj in range(a):
        xx = matlab_style_gauss2D((use_sz[0],use_sz[1]),1)
        xx = np.roll(xx,randshift[ii][jj] ,axis=randidx[ii][jj])+randomn[ii][jj]
        #xx = np.roll(xx,np.random.randint(-5,5) ,axis=np.random.randint(0,2))
        #xx += np.random.randn(1)
        response[:,:,ii,jj]=xx
    #response=np.expand_dims(response,3)
    [disp_row, disp_col, scale_ind,angle_ind]=resp_newton(response, use_sz)
    print ('disp_row:%f,disp_col:%f,scale_ind:%i,angle_ind:%i')%(disp_row,disp_col,scale_ind,angle_ind)
if __name__=="__main__":
    main()
