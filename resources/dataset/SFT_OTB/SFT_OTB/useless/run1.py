'''
Recurrent trackor
run.py
'''

from config import *
import pdb
####################################################
#####               Main Fun            ############
####################################################
if __name__ == '__main__':
    ####
    if not os.path.isdir(data_path):
        raise Exception('data not exist',data_path)

    #### 
    fdrs = next(os.walk(data_path))[1]
    fdrs = sorted(fdrs)
    n_fdr = len(fdrs)
    print("{}".format(n_fdr))
    #### build VGG model
    with tf.device(gpu_id):
        vgg = vgg19_tf.Vgg19( vgg_model_path, vgg_out_layers)
        vgg_sess = tf.Session(config = config)
    vgg_map_total, vgg_map_idx, vgg_map_nlayer = vgg.out_map_total, vgg.out_map_idx, vgg.out_map_nlayer
    
    ####
    for ifdr in idx_fdrs:#np.arange (start_sample,end_sample,step_sample):#n_fdr
	##
        fdr = fdrs[ifdr]
        #if fdr !='CarScale':# and fdr!='':# and fdr !='Coke':# and fdr!='Football':
        #    continue
        fpath = os.path.join(data_path,fdr)
        f_rcts = glob.glob(fpath+'/groundtruth_rect*.txt')
        f_imgs = glob.glob(fpath+'/*.jpg')
        n_img = len(f_imgs)
        n_rct = len(f_rcts)
        f_imgs = sorted(f_imgs,key=str.lower)

        print("{}:{}:{}".format(ifdr, fdr, n_img))
	## read images >> X0
        for ii in range(n_img):
            img = read_image(f_imgs[ii],True,True,-1)
            if ii == 0:
                im_sz = np.asarray([img.shape[1],img.shape[0]])
                X0 = np.zeros((n_img,img.shape[0],img.shape[1],3),dtype=np.uint8)
            X0[ii,:,:,:] = img
        del img		
	################# each sequence ##############################
        for iseq in range(n_rct):
            str1 = 'result_%s_%d_%.3f.mat' %(fdr,iseq,update_factor)
            fname = os.path.join(cache_path,str1)
            if os.path.isfile(fname):
                print("{} existed result".format(fname))
                continue
	
	    #### log file
            str1 = 'log_%s_%s_%d.txt' %(pstr,fdr,iseq) #pstr=gcnn
            log_file = os.path.join(cache_path,str1)
            logger = Logger(logname=log_file, loglevel=1, logger=">>").getlog()
	    
	    #### load rct and convert it ctr style 
            gt_rcts0 = np.loadtxt(f_rcts[iseq])
            gt_rcts = np.floor(fun_rct2ctr(gt_rcts0)) 
 
	    #### set peak map
            target_sz = gt_rcts[0,2:]
            window_sz,padding_h, padding_w = fun_get_search_window2(target_sz,im_sz,None,None) 

	    #### cell_size ??
            #xx = np.prod(np.floor(window_sz/cell_size0+0.5))
            #print('xx:{}'.format(xx))
            #cell_size = np.round(cell_size0*np.sqrt(xx/max_win2))
            cell_size=window_sz[0]/fea_sz[0]
            print("cell_size:{}".format(cell_size))

            #if xx > max_win2:
            #    cell_size = np.round(cell_size0*np.sqrt(xx/max_win2)+0.5)
            #elif xx < min_win2:
            #    cell_size = np.floor(cell_size0*np.sqrt(xx/min_win2)+0.5)
            #else:
            #    cell_size = cell_size0 # Bolt

	    #### 
            pmap,pmap_ctr_h,pmap_ctr_w = fun_get_peak_map(window_sz,target_sz,cell_size,fea_sz,False)
            assert(pmap.shape[0]==fea_sz[0] and pmap.shape[1]==fea_sz[1])
            str1 = "target_sz: [%d, %d], window_sz: [%d, %d], pmap.shape: [%d, %d], cellsize: [%d] " %(target_sz[0], target_sz[1],\
                     window_sz[0], window_sz[1], pmap.shape[0], pmap.shape[1],  cell_size)
            logger.info(str1)

            #### scale map
            #smap, swindow = fun_get_scale_map(sparam.nScales,sparam.scale_sigma_factor)

            #fea_sz = np.asarray([pmap.shape[1],pmap.shape[0]])
            prod_hw = fea_sz[1]*fea_sz[0]
            kernel_gamma2 = kernel_gamma/vgg_map_total
            #vgg_in_sz = fea_sz*vgg_scale_in2out
            pmap = pmap.reshape((-1,1)) 

	    #### cos_win
            cos_win = fun_cos_win(fea_sz[1],fea_sz[0],(1,fea_sz[1],fea_sz[0],1))*1.0e-3#/vgg_map_total

	    ####
            #vgg_map_total=3136
            vgg_map_af_total=vgg_map_total-vgg_map_conv1
            in_shape = (1,fea_sz[1],fea_sz[0],vgg_map_af_total)
            vgg_fea    = np.zeros(in_shape,dtype=np.float32) 


            #### graph 1
            gh1['height_width'] = (fea_sz[1],fea_sz[0])
            A1 = grid_graph(gh1, corners = False)
            L1 = laplacian(A1)

            del A1
	    ####
            pred_rcts  = np.zeros((n_img,4),dtype=np.float32)
            pred_rcts[0,:] = np.copy(gt_rcts[0,:])
            cur_rct = np.copy(pred_rcts[0,:])
	    
            save_fdr = '%s_%d' %(fdr,iseq)
            save_path = os.path.join(cache_path,save_fdr)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

	    ############################## extract feature ##########################
            #### other vars: padding, vgg_in_sz, padding_type
	    #### ims: [n,h,w,3] or [h,w,3]; ctr_rcts: [n,4]; vgg_fea: [n,nmap,nh,nw] 
            def extract_feature(ims,ctr_rcts,vgg_fea,cos_win,is_scale):
                n = ctr_rcts.shape[0]
                l = len(ims.shape)
                if l == 4:
                    assert(ims.shape[0]==n)
                ## crop out patch
                patches = []
                #pdb.set_trace()
                for ii in range(n):
                    window_sz,_,_ = fun_get_search_window2(ctr_rcts[ii,2:],None,padding_h,padding_w) # ??
                    if l==4: 
                        patch = fun_get_patch(np.copy(ims[ii]),ctr_rcts[ii,0:2],window_sz)
                    else:
                        patch = fun_get_patch(np.copy(ims),ctr_rcts[ii,0:2],window_sz)
                    patches.append(patch)
                ## extract vgg feature: outdata >> list
                patches = vgg_process_images(patches,**img_param)     
                feed_dict = {vgg.images: patches}
                if is_scale:
                    
                    vgg.out_layer=vgg.out_layers[0] 
                    outdata = vgg_sess.run(vgg.out_layer, feed_dict=feed_dict) 
                    outdata1=[]
                    outdata1.append(outdata)
                    #outdata1=[]
                    #for i in range(9):
                        #outdata1.append(np.expand_dims(outdata[i,:,:,:],axis=0))
                    
                    #vgg.out_layer=vgg.out_layers[1:]
                    #outdata1 = vgg_sess.run(vgg.out_layer, feed_dict=feed_dict)
                else:
                    vgg.out_layer=vgg.out_layers[1:]
                    outdata1 = vgg_sess.run(vgg.out_layer, feed_dict=feed_dict)

                vgg_resize_maps(outdata1,(fea_sz[1],fea_sz[0]), 'bicubic', vgg_fea)
                ## normalization of vgg feature ??
                if cos_win is not None:
                    vgg_fea[:,:,:,:] = vgg_fea*cos_win
                #if scale_win not None:
                #    vgg_fea[:,:,:,:] = vgg_fea*scale_win
                return patches
            
            #########################
            #ss = '_%s_%d' %(fdr,iseq)
            #nn['dir_name'] = create_dir(cache_path,ss)
            #nn['num_node'] = prod_hw
            #nn['Fin'] = vgg_map_total
            nn_d = nn_p*pca_energy #vgg_map_total
            nn_m = prod_hw
            assert(nn_d%nn_p ==0)
            nn_map = np.int32(nn_d/nn_p)
            ####
            target_sz_2=(1.0*fea_sz[0]/window_sz[0])*target_sz
            '''
            nn_K=np.round(max(target_sz_2)*1)
            if nn_K >fea_sz[0]*0.5:
                 nn_K=np.round(fea_sz[0]*0.5)#33.6
            ###
            '''
            nn_K=(max(target_sz_2)*1)
            if nn_K >fea_sz[0]*0.5:
                 nn_K=(fea_sz[0]*0.5)#33.6
            nn_K=np.round(nn_K/k_step)
            nn_K=nn_K.astype(np.int)
            im1 = np.copy(X0[0,:,:,:])
            nn_sK = np.int32(sparam_K_ratio * nn_K)
            idx_ctr = np.int32(np.floor(np.prod(fea_sz)/2))

            #### scale map
            #target_sz,nScales,scale_sigma_factor,scale_step,scale_model_max_area
            smap, scale_win,sfactor = fun_get_scale_map(target_sz,sparam_nScales,sparam_scale_sigma_factor, sparam_scale_step)
            smapf = np.fft.fft(smap,axis=0)
            smapf = np.reshape(smapf,(-1,1,1))
            #smapf.dtype=np.float32
            scale_win = np.reshape(scale_win,(-1,1,1))#n*1*1*1
            im_sz = np.array((im1.shape[1],im1.shape[0]))
            min_scale_factor, max_scale_factor = fun_compute_min_max_scale_factor(window_sz, target_sz, im_sz, sparam_scale_step)
            currentScaleFactor = 1



            str3="target_sz_2:[%d,%d],nn_K:[%d],im_sz:[w:%d,h:%d]"%(target_sz_2[0],target_sz_2[1],nn_K,im1.shape[1],im1.shape[0])
            logger.info(str3)
            with tf.device(gpu_id):
                 #### actual model
                 #mdl = cgcnn.cgcnn(**nn)
                 mdl_tr = graphtracker.GTTr(L1, nn_m, nn_d, nn_K, nn_p, nn_gamma)
                 mdl_te = graphtracker.GTTe(L1, nn_m, nn_d, nn_K, nn_p)    
                 #mdl_sc = graphtracker.GTGK(sparam_gamma,L1, nn_m, nn_d, nn_sK, nn_p)
                 sess_tr = tf.Session(config=config, graph = mdl_tr.graph) 
                 sess_te = tf.Session(config=config, graph = mdl_te.graph)
                 #sess_sc = tf.Session(config=config, graph = mdl_sc.graph)
	    ####
            flag_occ = 0
            for jj in range(n_img):
		#### sampling patch
                im = np.copy(X0[jj,:,:,:])

		#################################################################
		##################### predict process ###########################
		#################################################################
                if jj > 0:
                    #### location predition
                    tmp_rct = np.copy(cur_rct)
                    tmp_rct[2:] = target_sz*currentScaleFactor
                    wsz,_,_ = fun_get_search_window2(tmp_rct[2:],None,padding_h,padding_w)
                    patch_t = extract_feature(np.copy(im),np.expand_dims(tmp_rct,0),vgg_fea, cos_win,False)
                    vgg_fea2 = fea_pca_te(np.copy(vgg_fea[0]), nn_p, pca_projections) # m*d
                    feed_dict = {mdl_te.ph_data: vgg_fea2, mdl_te.ws: model_alpha}
                    pred = sess_te.run(mdl_te.pred,feed_dict=feed_dict)
                    resp = np.reshape(pred,(nn_p,fea_sz[1],fea_sz[0]))

                    if jj==1:
                        mxres0 = np.zeros(nn_p)
                        mx_hh = np.zeros(nn_p, dtype=np.int32)
                        mx_ww = np.zeros(nn_p, dtype=np.int32)
                    for ilayer in range(nn_p):
                        mxres0[ilayer],mx_hh[ilayer],mx_ww[ilayer] = get_max_ps(resp[ilayer,:,:],pmap_ctr_h,pmap_ctr_w)
                    mx_w,mx_h,mxres = compute_hw(mx_hh,mx_ww,mxres0)

                    #window_sz,_,_ = fun_get_search_window2(cur_rct[2:],None,padding_h,padding_w) # ??
                    cur_rct[0:2] = cur_rct[0:2] + 1.0*np.asarray([mx_w,mx_h])*wsz/fea_sz

                    #### ============ scale search ===============
                    count = 0
                    ctr0 = np.floor(cur_rct[0:2] + 0.5)
                    search_scale = sfactor * currentScaleFactor
                    for iscale in range(nscale):
                        tmp_rcts[count,0:2] = np.copy(ctr0)
                        tmp_rcts[count,2:]  = np.floor(search_scale[iscale]*target_sz+0.5) #np.floor(cur_rct[2:]*search_scale[iscale]+0.5) # 2016.12.11
                        count = count + 1
                    extract_feature(np.copy(im),tmp_rcts,test_vgg_fea_scale, None,True)  ## ??? *cos_win == None                  
                    vgg_fea2 = fea_pca_te_ms(np.copy(test_vgg_fea_scale), nn_p, pca_projections) # n*(h*w)*d2
                    vgg_fea2 =vgg_fea2 * scale_win
                    vgg_fea2_f = np.fft.fft(vgg_fea2,axis=0)
                    #vgg_fea2_f = np.reshape(vgg_fea2_f ,(nscale,-1,vgg_map_conv1))
                    #vgg_fea2_f = np.fft.fft(test_vgg_fea_scale,axis=0)
                    #vgg_fea2_f = np.reshape(vgg_fea2_f,(nscale,-1,vgg_map_conv1))
                    '''
                    vgg_fea2 = np.reshape(test_vgg_fea_scale,(nscale,-1,vgg_map_conv1))
                    vgg_fea2 = vgg_fea2 * scale_win # n*(h*w)*d2
                    vgg_fea2_f = np.fft.fft(vgg_fea2,axis=0)
                    '''
                    # sf_num= n*(h*w)*d2  vgg_fea2_f=n*(h*w)*d2 sf_den=n*1*1
                    #pdb.set_trace()
                    scale_response = np.real(np.fft.ifft(np.sum(sf_num*vgg_fea2_f,axis=(-2,-1),keepdims=True)/(sf_den+lamda),axis=0))
                    #scale_response = np.squeeze(scale_response)
                    #pdb.set_trace()
                    mx_scale_idx = np.argmax(scale_response)
                    #if mx_scale_idx<4:
                    #   mx_scale_idx+=4
                    #else:
                    #   mx_scale_idx-=4
                    #mx_scale_idx =np.argwhere(scale_response==np.max(scale_response.ravel()))[0]-1
                    '''
                    ## graph
                    for iscale in range(nscale):
                        feed_dict = {mdl_sc.ph_data: vgg_fea2[iscale]}
                        gfea = sess_sc.run(mdl_sc.xlist,feed_dict=feed_dict) # nexperts * m * d2
                        gfea = np.stack(gfea)
                        fea_ctr[iscale, :, :] = gfea[:,idx_ctr,:]
                    ##
                    resp_scale = fun_scale_model_te(fea_ctr, model_scale_w, sparam_m_type) # nexpert*nscale
                    mx_scale_idx = np.argmax(np.mean(resp_scale,axis=0)) 
                    '''
                    mx_scale = sfactor[mx_scale_idx]
                    currentScaleFactor = currentScaleFactor * mx_scale
                    if currentScaleFactor < min_scale_factor:
                        currentScaleFactor = min_scale_factor
                    elif currentScaleFactor > max_scale_factor:
                        currentScaleFactor = max_scale_factor
                    

                    ####
                    cur_rct[2:] = np.floor(target_sz*currentScaleFactor+0.5) # ??
                    cur_rct = fun_border_modification(cur_rct,im.shape[0],im.shape[1])
                    pred_rcts[jj,:] = cur_rct
                    #window_sz,_,_ = fun_get_search_window2(pred_rcts[jj,2:],None,padding_h,padding_w) # ??
                    #pred_rcts[jj,0:2] = cur_rct[0:2] + 1.0*np.asarray([mx_w,mx_h])*window_sz/fea_sz + search_offset[mx_offset,:]
                    #pred_rcts[jj,:] = fun_border_modification(np.copy(pred_rcts[jj,:]),im.shape[0],im.shape[1]) ## ????
                    #cur_rct = np.copy(pred_rcts[jj,:])

                    ####
                    str1 = "[%3d]:[%3.2f,%3.2f,%3.2f,%3.2f],[%3.2f,%3.2f,%3.2f,%3.2f],[%.4f,%.4f], [%d, %.3f]\n\t\t[%.4f,%.2f,%.2f,%d]\n\t\t%s\n\t\t%s\n\t\t%s\n\t\t%s" %(jj,\
                                      gt_rcts[jj,0],gt_rcts[jj,1],gt_rcts[jj,2],gt_rcts[jj,3],\
                                      pred_rcts[jj,0], pred_rcts[jj,1],pred_rcts[jj,2],pred_rcts[jj,3],\
                                      currentScaleFactor,mx_scale,flag_occ,update_factor,\
                                      mxres, mx_w, mx_h,mx_scale_idx,\
                                      vector2string(mx_ww,'float'),\
                                      vector2string(mx_hh,'float'),\
                                      vector2string(mxres0,'float'),\
                                      vector2string(scale_response,'float'))

                    logger.info(str1)
                    flag_occ = 0

                    str1 = '%04d.jpg' %(jj) #'T_%d.jpg'
                    fname = os.path.join(save_path,str1)
                    fun_draw_rct_on_image(X0[jj,:,:],fname,gt_rcts[jj,:],None,pred_rcts[jj,:])

                    str1 = 'T_%d_mask.jpg' %(jj)
                    fname = os.path.join(save_path,str1)

                    str1 = 'prop_tr_%s_%d_%d.mat' %(fdr,iseq,jj)
                    fname = os.path.join(save_path,str1)

 
		#################################################################
		########################### Preparing ###########################
		#################################################################
                patch_t = extract_feature(np.copy(im),np.expand_dims(cur_rct,0),vgg_fea, cos_win,False) 
                pdb.set_trace()
                if jj == 0: # pca
                     pca_projections = fea_pca_tr(np.copy(vgg_fea[0]), nn_p, pca_energy, pca_is_mean, pca_is_norm)
                     
                vgg_fea2 = fea_pca_te(np.copy(vgg_fea[0]), nn_p, pca_projections)
 
		################### update model  ###################################
                if (jj%cf_nframe_update == 0 or jj < 5): 
                    with tf.device(gpu_id):
                        feed_dict = {mdl_tr.ph_data: vgg_fea2, mdl_tr.ph_labels: pmap}
                        ws = sess_tr.run([mdl_tr.ws],feed_dict)
                        #print("ws_shape:{}".format(ws.shape))
                        ws=np.array(ws[0]) 
                        if jj==0:
                            model_alpha = np.copy(ws)
                            model_x =np.copy(vgg_fea2)
                        else:
                            if np.isnan(np.max(np.absolute(ws))) == False:
                                model_alpha = (1-update_factor)*model_alpha + update_factor*ws
                                model_x     = (1-update_factor)*model_x    + update_factor*vgg_fea2
                                #falg_occ = 1
                            else:
                                flag_occ = 1


                        #### ========= scale =================
                        if jj == 0:
                            nscale = sparam_nScales
                            tmp_rcts = np.zeros((nscale,4))
                            fea_ctr = np.zeros((nscale, nn_p,  nn_sK*pca_energy), dtype = np.float32)
                        count = 0
                        ctr0 = np.floor(cur_rct[0:2] + 0.5)
                        search_scale = sfactor * currentScaleFactor 
                        for iscale in range(nscale):
                            tmp_rcts[count,0:2] = np.copy(ctr0)
                            tmp_rcts[count,2:]  = np.floor(target_sz*search_scale[iscale]+0.5) # 2016.12.11
                            count = count + 1
                        if jj == 0:
                            test_vgg_fea_scale  = np.zeros((nscale,in_shape[1],in_shape[2],3072),dtype=np.float32)
                            print("{}:{}".format( search_scale, nscale))
                        extract_feature(np.copy(im),tmp_rcts,test_vgg_fea_scale, None,True)  ## ??? *cos_win == None                  
                        vgg_fea2 = fea_pca_te_ms(np.copy(test_vgg_fea_scale), nn_p, pca_projections) # n*(h*w)*d2
                        #pdb.set_trace()
                        vgg_fea2 =vgg_fea2 * scale_win
                        vgg_fea2_f = np.fft.fft(vgg_fea2,axis=0)
                        #vgg_fea2_f = np.reshape(vgg_fea2_f ,(nscale,-1,vgg_map_conv1))
                        
                        '''
                        vgg_fea2 = np.reshape(test_vgg_fea_scale,(nscale,-1,vgg_map_conv1))
                        vgg_fea2 =vgg_fea2 * scale_win # n*(h*w)*d2
                        vgg_fea2_f = np.fft.fft(vgg_fea2,axis=0)
                        '''
                        new_sf_num = smapf * np.conj(vgg_fea2_f)#n*(h*w)*d2
                        new_sf_den = np.sum(vgg_fea2_f * np.conj(vgg_fea2_f),axis=(-2,-1),keepdims=True)#n*1

                        ## graph
                        '''
                        for iscale in range(nscale):
                            feed_dict = {mdl_sc.ph_data: vgg_fea2[iscale]}
                            gfea = sess_sc.run(mdl_sc.xlist,feed_dict=feed_dict) # nexperts * m * d2
                            #pdb.set_trace()
                            gfea = np.stack(gfea)
                            fea_ctr[iscale, :, :] = gfea[:,idx_ctr,:]              
                        ## model
                        ws = fun_scale_model_tr(fea_ctr, smap, sparam_gamma, sparam_m_type) 
                        '''
                        if jj==0:
                            #model_scale_w = np.copy(ws)
                            #model_x =np.copy(vgg_fea2)
                            sf_num = new_sf_num
                            sf_den = new_sf_den
                        else:
                            #pdb.set_trace()
                            if np.isnan(np.max(np.absolute(sf_den))) == False:
                               sf_num = (1-update_factor_s)*sf_num + update_factor_s*new_sf_num
                               sf_den = (1-update_factor_s)*sf_den + update_factor_s*new_sf_den
                            #falg_occ = 1
                            else:
                                flag_occ = 1

                       #### ================================= 
	    ## save all results
            sess_tr.close()
            sess_te.close()
            #sess_sc.close()
            if jj==n_img-1:
                pcs_loc_mean,pcs_loc_curv, pcs_loc_diff = fun_precision_location(pred_rcts,gt_rcts)
                pcs_olp_mean,pcs_olp_curv, pcs_olp_diff = fun_precision_overlap(pred_rcts,gt_rcts)
                str1 = '[%s, %d, %.3f]--[%.4f,%.4f]\n' %(fdr,iseq, update_factor, pcs_loc_mean,pcs_olp_mean),\
                                        np.array2string(pcs_loc_curv.transpose(),precision=4,separator=', '),\
                                        np.array2string(pcs_olp_curv.transpose(),precision=4,separator=', ') 
                logger.info(str1)
                close_logger(logger)

                str1 = 'result_%s_%d_%.3f.mat' %(fdr,iseq,update_factor)
                fname = os.path.join(cache_path,str1)
                save_mat_file(fname,gt_rcts,pred_rcts,pcs_loc_diff,pcs_olp_diff)
 
    vgg_sess.close()
     