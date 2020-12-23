'''
Recurrent trackor
runFast.py
'''

from config import *
#from funRT2 import *
#from strnn_conv import STRNN

####################################################
#####               Main Fun            ############
####################################################
if __name__ == '__main__':
    ####
    if not os.path.isdir(data_path):
        raise Exception('data not exist',data_path)

    #### 
    fdrs = next(os.walk(data_path))[1]
    n_fdr = len(fdrs)
    print (n_fdr)


    #### build VGG model
    vgg_net = VGGInferNet(vgg_model_path,vgg_model_type,vgg_out_layers,vgg_is_lrn)
    vgg_map_nums  = vgg_net.mapnums
    vgg_map_total = np.sum(vgg_map_nums[vgg_out_layers-1])

    ####
    for ifdr in np.arange(42,n_fdr):#(n_fdr):
	####
        fdr = fdrs[ifdr]
	#if fdr !='Skiing':# and fdr !='Coke':# and fdr!='Football':
	#    continue
        fpath = os.path.join(data_path,fdr)
        f_rcts = glob.glob(fpath+'/groundtruth_rect*.txt')
        f_imgs = glob.glob(fpath+'/*.jpg')
        n_img = len(f_imgs)
        n_rct = len(f_rcts)
        f_imgs = sorted(f_imgs,key=str.lower)

        print (n_img)	
	#### read images >> X0: n*h0*w0*3
        for ii in xrange(n_img):
            img = read_image(f_imgs[ii],True,True,-1)
            if ii == 0:
                im_sz = np.asarray([img.shape[1],img.shape[0]])
                X0 = np.zeros((n_img,img.shape[0],img.shape[1],3),dtype=np.uint8)
            X0[ii,:,:,:] = img
            del img		

	################# each sequence ##############################
        print (fpath,n_rct)
        for iseq in xrange(n_rct):
	    #### log file
            str1 = 'log_%s_%s_%d.txt' %(pstr,fdr,iseq)
            log_file = os.path.join(cache_path,str1)
            logger = Logger(logname=log_file, loglevel=1, logger=">>").getlog()

	    #### load rct and convert it ctr style 
	    #### vars: gt_rcts >> n*[w_ctr,h_ctr,w_len,h_len]
            gt_rcts0 = np.loadtxt(f_rcts[iseq])
            gt_rcts = np.floor(fun_rct2ctr(gt_rcts0)) # [w_ctr,h_ctr,w_len,h_len]
 
	    #### peak map
	    #### vars: window_sz >> [wsize,hsize];  padding_type; pmap;
	    ####       batch_y   >> n*1*h*w      ;  fea_sz       
            target_sz = gt_rcts[0,2:]
            window_sz,padding_type = fun_get_search_window(target_sz,im_sz,padding,None)  
            pmap,pmap_ctr_h,pmap_ctr_w = fun_get_peak_map(window_sz,target_sz,cell_size,False)
            print ('target_sz:',target_sz,';window_sz:', window_sz,';pmap,shape,ctr:',pmap.shape,pmap_ctr_h,pmap_ctr_w)
	
            fea_sz = np.asarray([pmap.shape[1],pmap.shape[0]])
            y = np.expand_dims(pmap,axis=0)
            y = np.expand_dims(y,axis=0)
            yf = np.fft.fft2(y,axes=(-2,-1))#fun_get_freq_fea(batch_y,False)
            srnn_step = np.min(np.int32(fea_sz/5))
            print ('srnn_step=',srnn_step)

	    #### vars: vgg_in_sz;   vgg_in_sz_aug;    window_sz_aug;
            vgg_in_sz     = fea_sz * vgg_scale_in2out
            vgg_in_sz_aug = vgg_in_sz + 2*srnn_step * vgg_scale_in2out
            fea_sz_aug    = fea_sz + 2*srnn_step

            def compute_window_sz_aug(window_sz,vgg_in_sz,srnn_step,vgg_scale_in2out):
                window_sz_aug = np.floor(window_sz + np.divide(1.*window_sz,vgg_in_sz)*2*srnn_step*vgg_scale_in2out+0.5) ## augmented sz
                return window_sz_aug
            window_sz_aug = compute_window_sz_aug(window_sz,vgg_in_sz,srnn_step,vgg_scale_in2out)
	     
	    #### cos_win
            cos_win = fun_cos_win(fea_sz_aug[1],fea_sz_aug[0],(1,1,fea_sz_aug[1],fea_sz_aug[0]))
	    #### vars: pred_rcts; save_path; cur_rct
            pred_rcts  = np.zeros((n_img,4),dtype=np.float32)
            pred_rcts[0,:] = np.copy(gt_rcts[0,:])
            cur_rct = np.copy(pred_rcts[0,:])
	    
            save_fdr = '%s_%d' %(fdr,iseq)
            save_path = os.path.join(cache_path,save_fdr)
            if not os.path.isdir(save_path):
               os.mkdir(save_path)
		
	    #### vars: cur_fea_aug >> 1*(nd*c)*h_aug*w_aug;  batch_fea:  n*(nd*c)*h*w
            strnn_list,trnn_flag = fun_get_strnn_list(srnn_directions)
            strnn_ntracker = len(strnn_list)
            strnn_ndirection = len(strnn_list[0])
	    
            cur_cf_fea        = np.zeros((1,strnn_ndirection*vgg_map_total,fea_sz[1],fea_sz[0]),dtype=np.float32)
            #all_cf_fea            = np.zeros((strnn_ntracker,strnn_ndirection*vgg_map_total,fea_sz[1],fea_sz[0]),dtype=np.float32)

            vgg_fea_aug       = np.zeros((1,vgg_map_total,fea_sz_aug[1],fea_sz_aug[0]),dtype=np.float32)
            vgg_fea_aug_tm1       = np.zeros((1,vgg_map_total,fea_sz_aug[1],fea_sz_aug[0]),dtype=np.float32)
            #batch_fea_aug     = np.zeros((batch_size,vgg_map_total,fea_sz_aug[1],fea_sz_aug[0]),dtype=np.float32)
            #batch_fea_aug_tm1 = np.zeros((batch_size,vgg_map_total,fea_sz_aug[1],fea_sz_aug[0]),dtype=np.float32)

            model_alphaf = np.zeros((strnn_ntracker,fea_sz[1],fea_sz[0]),dtype=np.complex64)
            model_xf = np.zeros((strnn_ntracker,1,strnn_ndirection*vgg_map_total,fea_sz[1],fea_sz[0]),dtype=np.complex64)

	    ####
            history_res  = np.zeros((n_historical_sample+1,strnn_ntracker),dtype=np.float32)
            history_loss = np.zeros((n_historical_sample+1,strnn_ntracker),dtype=np.float32)
            score_tracker = np.ones((strnn_ntracker),dtype=np.float32)
            score_tracker[-1] = 0
	    ############################## extract feature ##########################
	    #### other vars: padding, vgg_in_sz, srnn_step, window_sz, vgg_in_sz, vgg_scale_in2out, padding_type,vgg_in_sz_aug, vgg_fea_aug
	    #### output: vgg_fea_aug >> 1*nmap*h_aug*w_aug
            def extract_feature(im,ctr_rct,vgg_fea_aug):
		## crop out patch
		#print ctr_rct
                window_sz,_ = fun_get_search_window(ctr_rct[2:],None,padding,padding_type)
                window_sz_aug = compute_window_sz_aug(window_sz,vgg_in_sz,srnn_step,vgg_scale_in2out)
                patch = fun_get_patch(np.copy(im),ctr_rct[0:2],window_sz_aug)
		#print window_sz_aug,window_sz
                #patch = get_patch_warping(np.copy(im),ctr_rct[0:2],window_sz_aug,vgg_in_sz_aug)## window_sz will change ??
                patch = np.expand_dims(patch,axis=0) # 1*h*w*3
		## extract vgg feature: outdata >> list
                outdata,CLASSES = vgg_net.get_output_batch(patch,vgg_in_sz_aug[1],vgg_in_sz_aug[0],'keep_all_content')
		## from list to array
		#t0 = time.time()
                fun_vggfea_list2array(outdata,fea_sz_aug[1],fea_sz_aug[0],'bicubic',True,vgg_fea_aug)
		#print 'list2array:',time.time()-t0
		## normalization of vgg feature ??
		#nm = np.sqrt((np.square(np.square(vgg_fea_aug))).sum(axis=1,keepdims=True))
                vgg_fea_aug[:,:,:,:] = vgg_fea_aug*(cos_win*1.0e-4)
                return patch[0]
	
	    ####  for each frame ####
            for jj in xrange(n_img):
                im = np.copy(X0[jj,:,:,:])

		#################################################################
		##################### predict process ###########################
		#################################################################
                if jj > 0:
		    ###########################
		    ###### filter predict #####
		    ###########################
                    wsz,_ = fun_get_search_window(cur_rct[2:],None,padding,padding_type)
                    search_offset = fun_get_search_ctr(wsz,factor=0.4)
		    #print cur_rct,wsz,search_offset
                    noffset      = search_offset.shape[0]

                    nscale = len(search_scale)	
                    tmp_rct = np.zeros_like(cur_rct)
                    response = np.zeros((noffset,strnn_ntracker,nscale,fea_sz[1],fea_sz[0]))

		    #############################
                    flag_detect = True
                    for ioffset in xrange(noffset):
                        tmp_rct[0:2] = search_offset[ioffset,0:2] + cur_rct[0:2]
                        for iscale in range(nscale):
                            tmp_rct[2:] = cur_rct[2:]*search_scale[iscale]
                            extract_feature(np.copy(im),tmp_rct,vgg_fea_aug) # vgg_fea_aug >> 1*nmap*h_aug*w_aug
                            for itracker in np.arange(strnn_ntracker-1,-1,-1):
                                fun_shift_feas(vgg_fea_aug,vgg_fea_aug_tm1,strnn_list[itracker],\
							 srnn_step, cur_cf_fea) # cur_fea_ndt >> 1*(nd*c)*h*w
                                cur_zf = np.fft.fft2(cur_cf_fea,axes=(-2,-1))#fun_get_freq_fea(cur_fea_ndt,True) # 1*(nd*c)*h*w
                                response[ioffset,itracker,iscale,:,:] = fun_response(model_alphaf[itracker],\
							model_xf[itracker],cur_zf,kernel_type,kernel_sigma,kernel_gamma)
				#if flag_detect == True and ioffset!=0:
				#    break
				 
                        if ioffset == 0: ## no offset
                            mx_scale = get_scale(response[ioffset])
                            mxres_tracker,mxres_tracker_hw,(mx_h,mx_w),mx_nd = \
						get_ps_offset(response[ioffset,:,mx_scale,:,:],pmap_ctr_h,pmap_ctr_w,score_tracker) #
			    ## decision whether searching needs 
                            flag_detect = fun_is_detect(history_res,mxres_tracker) 

                        if not flag_detect or jj < n_historical_sample:
                            mx_offset = ioffset
                            flag_detect = False
                            break
                    if flag_detect: ## if detecting
                        mx_offset = get_ctr_shift_trackers(response)
                        mx_scale = get_scale(response[mx_offset])
                        ## compute ps offset 
                        mxres_tracker,mxres_tracker_hw,(mx_h,mx_w),mx_nd = \
						get_ps_offset(response[mx_offset,:,mx_scale,:,:],pmap_ctr_h,pmap_ctr_w,score_tracker) #
                        print (mx_h,mx_w,mx_offset)

		    ######################
                    if jj < n_historical_sample:
                        mx_nd = -1
                        mx_h = mxres_tracker_hw[mx_nd,0]
                        mx_w = mxres_tracker_hw[mx_nd,1]

		    ####		
                    update_idx = np.mod(jj,n_historical_sample)
                    history_res[update_idx,:] = np.copy(mxres_tracker)
                    score_tracker = find_stable_tracker(response[mx_offset,:,mx_scale,:,:],\
						mxres_tracker,mx_h,mx_w,history_loss,update_idx)#history_loss changes
                    #history_loss[idx,:] = loss0[:,:]

		    ####
		    #mx_h = search_offset[mx_offset,1] + mx_h #- pmap_ctr_h
                    #mx_w = search_offset[mx_offset,0] + mx_w #- pmap_ctr_w

                    #### compute position ?
                    pred_rcts[jj,2:] = np.floor(cur_rct[2:]*search_scale[mx_scale]+0.5)
                    window_sz,_ = fun_get_search_window(pred_rcts[jj,2:],None,padding,padding_type)
                    pred_rcts[jj,0:2] = cur_rct[0:2] + 1.0*np.asarray([mx_w,mx_h])*window_sz/fea_sz + search_offset[mx_offset,:]
                    pred_rcts[jj,:] = fun_border_modification(np.copy(pred_rcts[jj,:]),im.shape[0],im.shape[1])
                    cur_rct = np.copy(pred_rcts[jj,:])
		  
		    #### 
                    str1 = '[%3d]:[%3.2f,%3.2f,%3.2f,%3.2f],[%3.2f,%3.2f,%3.2f,%3.2f],[%s,%s],[%.2f,%.2f],[%s],[%d,%d]' %(jj,gt_rcts[jj,0],\
						gt_rcts[jj,1],gt_rcts[jj,2],gt_rcts[jj,3],\
                                                pred_rcts[jj,0], pred_rcts[jj,1],pred_rcts[jj,2],pred_rcts[jj,3],\
						np.array2string(mxres_tracker,precision=4),\
						np.array2string(score_tracker,precision=4),mx_h,mx_w,srnn_directions[mx_nd],flag_detect,mx_offset)
                    logger.info(str1)

                    str1 = 'T_%04d.jpg' %(jj)
                    fname = os.path.join(save_path,str1)
                    fun_draw_rct_on_image(im,fname,gt_rcts[jj,:],None,pred_rcts[jj,:])
              
 
		#### **************  extract fea ****************************
                patch = extract_feature(np.copy(im),cur_rct,vgg_fea_aug) # vgg_fea_aug >> 1*nmap*h_aug*w_aug
                str1 = 'vgg_fea_aug_%d' %(jj)
                fname = os.path.join(save_path,str1)
                #save_vars6_dumps(fname,vgg_fea_aug,None,None,None,None,None)
		#save_mat_file(fname,patch,None,None,None)

		#### ************* update model  ****************************
               	if (jj%srnn_nframe_update == 0 or jj<5):
		    #if trnn_flag==True and jj==0:
			#vgg_fea_aug_tm1[:,:,:,:] = vgg_fea_aug[:,:,:,:]
                    for kk in xrange(strnn_ntracker):
                        fun_shift_feas(vgg_fea_aug,vgg_fea_aug_tm1,strnn_list[kk],srnn_step, cur_cf_fea)
		    	#### strnn Update ####
                        xf = np.fft.fft2(cur_cf_fea,axes=(-2,-1)) # 1*c*h*w
                        alphaf = fun_w(xf,yf,kernel_type,kernel_sigma,kernel_gamma) # h*w
                        if jj==0:
                            model_alphaf[kk] = np.copy(alphaf)
                            model_xf[kk]     =  np.copy(xf)
                        else:
                            model_alphaf[kk] = (1-update_factor)*model_alphaf[kk] + update_factor*alphaf 	
                            model_xf[kk]     = (1-update_factor)*model_xf[kk]     + update_factor*xf
                if trnn_flag==True:
                    vgg_fea_aug_tm1[:,:,:,:] = vgg_fea_aug[:,:,:,:]

	    ## save all results
            if jj==n_img-1:
                pcs_loc_mean,pcs_loc_curv, pcs_loc_diff = fun_precision_location(pred_rcts,gt_rcts)
                pcs_olp_mean,pcs_olp_curv, pcs_olp_diff = fun_precision_overlap(pred_rcts,gt_rcts)
                str1 = '[%s, %d]--[%.4f,%.4f]\n' %(fdr,iseq,pcs_loc_mean,pcs_olp_mean),\
					np.array2string(pcs_loc_curv.transpose(),precision=4,separator=', '),\
					np.array2string(pcs_olp_curv.transpose(),precision=4,separator=', ') 
                logger.info(str1) 
                close_logger(logger)

                str1 = 'result_%s_%d.mat' %(fdr,iseq)
                fname = os.path.join(cache_path,str1)
                save_mat_file(fname,gt_rcts,pred_rcts,pcs_loc_diff,pcs_olp_diff)

     

     
