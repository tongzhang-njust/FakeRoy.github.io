'''
Recurrent trackor
run.py
'''

from config_final import *
import pdb
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

    #### build VGG model
    #if True: #with tf.device(gpu_id):
    vgg = vgg19_tf2.Vgg19(vgg_model_path, vgg_out_layers, sparam_nAngles, vgg_out_layers2)
    vgg_sess = tf.Session(config = config)
    vgg_map_total, vgg_map_idx, vgg_map_nlayer = vgg.out_map_total, vgg.out_map_idx, vgg.out_map_nlayer
    vgg_map_total2, vgg_map_idx2, vgg_map_nlayer2 = vgg.out_map_total2, vgg.out_map_idx2, vgg.out_map_nlayer2
    
    ########################################################
    for ifdr in idx_fdrs:
        fdr = fdrs[ifdr]
         
        #if fdr != 'book' and fdr != 'bag':
        #    continue

        fpath = os.path.join(data_path,fdr)
        f_rcts = glob.glob(fpath+'/groundtruth_rect*.txt')
        f_imgs = glob.glob(fpath+'/img/*.jpg')
        n_img = len(f_imgs)
        n_rct = len(f_rcts)
        f_imgs = sorted(f_imgs,key=str.lower)
        print("{}:{}:{}".format(ifdr, fdr, n_img))

        #### read images >> X0
        n_img = 30
        for ii in range(n_img):
            img = read_image(f_imgs[ii],True,True,-1)
            if ii == 0:
                im_sz = np.asarray([img.shape[1],img.shape[0]])
                X0 = np.zeros((n_img,img.shape[0],img.shape[1],3),dtype=np.uint8)
            X0[ii,:,:,:] = img
        del img	
	
        #################### each sequence ############################
        for iseq in range(n_rct):
            str1 = 'result_%s_%d_%.3f.mat' %(fdr,iseq,update_factor)
            fname = os.path.join(cache_path,str1)
            if os.path.isfile(fname):
                print("{} existed result".format(fname))
                continue
	
            ####
            str1 = 'log_%s_%s_%d.txt' %(pstr,fdr,iseq) #pstr=gcnn
            log_file = os.path.join(cache_path,str1)
            logger = Logger(logname=log_file, loglevel=1, logger=">>").getlog()

            save_fdr = '%s_%d' %(fdr,iseq)
            save_path = os.path.join(cache_path,save_fdr)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
	   
            #### load original contour file
            gt_rcts0 = np.loadtxt(f_rcts[iseq],delimiter=',')
            gt_rcts = np.floor(fun_rct2ctr(gt_rcts0))
            #gt_contours = np.loadtxt(f_rcts[iseq])
 
            #### << ctr variables >> ####
            target_sz = gt_rcts[0,2:]
            window_sz, padding_h, padding_w = fun_get_search_window2(target_sz,im_sz,None,None) ## ??? may be changed
 
            cell_size=window_sz[0]/fea_sz[0] 
            prod_hw = fea_sz[1]*fea_sz[0]

            #### peak map
            # pdb.set_trace()
            pmap,pmap_ctr_h,pmap_ctr_w = fun_get_peak_map(window_sz,target_sz,cell_size,fea_sz,False)
            assert(pmap.shape[0]==fea_sz[0] and pmap.shape[1]==fea_sz[1])
            str1 = "target_sz: [%d, %d], window_sz: [%d, %d], pmap.shape: [%d, %d], cellsize: [%d] " %(target_sz[0], target_sz[1],\
                     window_sz[0], window_sz[1], pmap.shape[0], pmap.shape[1],  cell_size)
            logger.info(str1)
            pmap = pmap.reshape((prod_hw,1))

	        #### cos_win
            cos_win = fun_cos_win(fea_sz[1],fea_sz[0],(1,fea_sz[1],fea_sz[0],1))*1.0e-3#/vgg_map_total

            ############################# extract feature ##########################
            #### ims: [n,h,w,3]; vgg_fea: [n,nh,nw, nchannel] 
            def extract_feature(ims, cos_win = None, vgg_out_norm_hw = (28,28), vgg_ind = 1):
                vgg_out_norm_h, vgg_out_norm_w = vgg_out_norm_hw[0], vgg_out_norm_hw[1]

                ## extract vgg feature: outdata >> list
                if vgg_ind == 1: 
                    feed_dict = {vgg.images:ims, vgg.norm_h:vgg_out_norm_h, vgg.norm_w:vgg_out_norm_w}
                    vgg_fea = vgg_sess.run(vgg.vgg_fea, feed_dict=feed_dict) 
                elif vgg_ind == 2:
                    feed_dict = {vgg.images:ims, vgg.norm_h2:vgg_out_norm_h, vgg.norm_w2:vgg_out_norm_w}
                    vgg_fea = vgg_sess.run(vgg.vgg_fea2, feed_dict=feed_dict)
                else:
                    assert(vgg_ind==1 or vgg_ind==2) 
                if cos_win is not None:
                    vgg_fea = vgg_fea*cos_win

                return vgg_fea
            
            def compute_win_szs(target_szs, padding_hw):
                padding_h, padding_w = padding_hw[0], padding_hw[1]
                win_szs = np.zeros_like(target_szs)
                n = win_szs.shape[0]
                for ii in range(n):
                    win_szs[ii,:], _ ,_ = fun_get_search_window2(target_szs[ii,:],None,padding_h,padding_w)
                return win_szs 
            
            ###################
            #### << PCA >> ####
            ###################
            nn_m = prod_hw
            nn_p = vgg_map_nlayer
            nn_p2 = vgg_map_nlayer2
            if pca_flag:
                nn_d = nn_p*pca_energy #vgg_map_total
                nn_d2 = nn_p2*pca_energy
                assert(nn_d%nn_p ==0)
                nn_map_idx = np.arange(0,nn_d+1, pca_energy)
                nn_map_idx2 = np.arange(0,nn_d2+1, pca_energy)
            else:
                nn_d = vgg_map_total
                nn_d2 = vgg_map_total2
                nn_map_idx = vgg_map_idx
                nn_map_idx2 = vgg_map_idx2

            ###################
            #### << K >> ######
            ###################
            target_sz_within_fea = (1.0*fea_sz[0]/window_sz[0])*target_sz
            nn_K=(max(target_sz_within_fea)*1)
            if nn_K >fea_sz[0]*0.5:
                nn_K = fea_sz[0]*0.5#33.6
            nn_K=np.round(nn_K/k_step)
            nn_K=nn_K.astype(np.int32)

            im_sz = np.array((X0.shape[2],X0.shape[1])) 
            str3="target_sz_within_fea:[%d,%d],nn_K:[%d],im_sz:[w:%d,h:%d]" %(target_sz_within_fea[0],target_sz_within_fea[1],nn_K,im_sz[0],im_sz[1])
            logger.info(str3)

            #########################
            #### << scale factor>> ##
            #########################
            scale_factors, min_scale_factor, max_scale_factor = get_scale_factors(sparam_nScales, sparam_scale_step, im_sz, target_sz, window_sz)
            #min_scale_factor, max_scale_factor = fun_compute_min_max_scale_factor(window_sz, target_sz, im_sz, sparam_scale_step)
            currentScaleFactor = 1.0 #np.asarray([1,1])
            currentAngleFactor = 0.0
            str3="min/max scale factor:[%.4f,%.4f]" %(min_scale_factor, max_scale_factor)
            logger.info(str3)

            ###############################
            ##### << graph tracker >> #####
            ###############################
            L1=sio.loadmat('L1.mat')
            L1=L1['data1']
            L1=L1.tocsr()
            #save_vars6_dumps('L1_20171112.pkl', L1, None, None, None, None, None)
            #L1=pickle.load(open('L.pkl','rb'))
            mdl_tr = graphtracker.GTTr(L1, nn_m, nn_map_idx, nn_K, nn_gamma, fast_max_iter = 5)
            sess_tr = tf.Session(config=config, graph = mdl_tr.graph)
            mdl_te = graphtracker.GTTe(L1, nn_m, nn_map_idx, nn_K)    
            sess_te = tf.Session(config=config, graph = mdl_te.graph)
            
            ## scale
            #idx_wgt_scale = find_weight_idx(nn_map_idx,vgg_out_layers,vgg_out_layers2,nn_K)
            mdl_tr2 = graphtracker.GTTr(L1, nn_m, nn_map_idx2, nn_K, nn_gamma, fast_max_iter = 5)
            sess_tr2 = tf.Session(config=config, graph = mdl_tr2.graph)
            mdl_te2 = graphtracker.GTTe(L1, nn_m, nn_map_idx2, nn_K)
            sess_te2 = tf.Session(config=config, graph = mdl_te2.graph)

            ###############################
            #xx = np.int32(np.max(target_sz))
            scale_window_sz = window_sz #np.floor((window_sz+xx)/2.) # cuizhen ?? np.asarray([xx,xx], dtype = np.float32)
            # pdb.set_trace()
            index = np.mod(scale_window_sz,2)==0
            scale_window_sz[index] = scale_window_sz[index] + 1
            idx_ctr = np.int32(np.floor(np.prod(fea_sz)/2))

            pred_rcts  = np.zeros((n_img,4),dtype=np.float32)
            pred_rcts[0,:] = np.copy(gt_rcts[0,:])

            #pred_contours  = np.zeros((n_img,8),dtype=np.float32)
            #pred_contours[0,:] = np.copy(gt_contours[0,:])

            cur_rct = np.copy(pred_rcts[0,:])
            #cur_contour = np.copy(pred_contours[0,:])

            ###########################################
            ############ Tracking #####################
            ###########################################
            for jj in range(n_img):
                im = np.copy(X0[jj,:,:,:])

            ###############################
            #### << predict process >> ####
            ###############################
                if jj > 0:
                    ####################################################
                    ###### ctr prediction + scale/angle search ##########################
                    ####################################################
                    ## use target_sz

                    old_tgt_ctr = np.floor(cur_rct[0:2] + 0.5) # [w,h], float32
                    # old_bind: int32, others are float32, [h, w]
                    old_bbx_hw, old_bind, new_img_sz, new_img_offset, new_bbx_hw, new_bbx_hw_ratio \
                                = get_bounding_boxes_for_nscale(old_tgt_ctr, scale_window_sz, currentScaleFactor, scale_factors, padFactor=1.2) # [h,w]
                    pdb.set_trace()
                    new_im = fun_get_patch(np.copy(im), old_tgt_ctr, new_img_sz[::-1])
                    #vgg.nangle = sparam_nAngles

                    feed_dict = {vgg.imgs: np.expand_dims(new_im,0), \
                                       vgg.angles: currentAngleFactor + sparam_angles, \
                                       vgg.img_sz: new_img_sz,     vgg.bbx: new_bbx_hw_ratio,         vgg.bind: old_bind, \
                                       vgg.crop_size: np.asarray([224,224], dtype=np.int32)}


                    ## smp_patches: [al, sl, 224, 224, c]; smp_transforms: list of len=al, [8]; smp_offset: [2]; all them [h,w]
                    smp_patches, smp_transforms = vgg_sess.run([vgg.patches, vgg.transforms], feed_dict=feed_dict)

                    smp_patches = np.reshape(smp_patches,(-1,224,224,3))
                    vgg_feas = extract_feature(smp_patches, cos_win = cos_win, vgg_out_norm_hw = fea_sz, vgg_ind = 2) 
                    
                    # [al*sl, fea_sz, nlayer*ch]                    
                    vgg_feas = np.reshape(vgg_feas,(sparam_nAngles, sparam_nScales, fea_sz[1], fea_sz[0], vgg_map_total2)) 

                    ## compute responses for rotating and scaling features on multiple layers
                    responses = np.zeros((sparam_nAngles, sparam_nScales, nn_p2, fea_sz[1], fea_sz[0]))
                    #model_alpha_scale = model_alpha[idx_wgt_scale,:]
                    for iangle in range(sparam_nAngles):
                        for iscale in range(sparam_nScales):
                            fea = np.reshape(vgg_feas[iangle,iscale],(nn_m,nn_d2))

                            feed_dict = {mdl_te2.ph_data: fea, mdl_te2.ws: model_alpha2}
                            pred = sess_te2.run(mdl_te2.pred, feed_dict=feed_dict)
                            responses[iangle,iscale,:,:,:] = np.reshape(pred,(nn_p2,fea_sz[1],fea_sz[0]))

                    ## ======= newton =============
                    pos_wh = np.zeros((nn_p2,2), dtype=np.float32)
                    mx_scales = np.zeros(nn_p2, dtype=np.float32)
                    mx_angles = np.zeros(nn_p2, dtype=np.float32)
                    max_response = np.zeros((sparam_nAngles, sparam_nScales, nn_p2), dtype=np.float32)
                    #rect_points = np.zeros((nn_p2,4,2), dtype=np.float32)
                    for ilayer in range(nn_p2):
                        mx_angle_idx, mx_scale_idx, mx_offset_h, mx_offset_w, max_response[:,:,ilayer] = \
                                                 resp_newton(responses[:,:,ilayer,:,:], fea_sz, iterations = 5)

                        ## new ctr
                        new_ctr_h, new_ctr_w = np.floor(new_img_sz/2)

                        ## ctr
                        bbx = new_bbx_hw[mx_scale_idx,:] # [h1,w1,h2,w2]
                        in_sz = np.asarray([bbx[3]-bbx[1], bbx[2]-bbx[0]]) # [w,h]
                        mx_offset_wh = 1.0*np.asarray([mx_offset_w,mx_offset_h])*in_sz/fea_sz # [w,h]
                        pos_h, pos_w = mx_offset_wh[1] + new_ctr_h, mx_offset_wh[0] + new_ctr_w

                        pos_wh[ilayer,:] = get_ori_coordinates(pos_h, pos_w, smp_transforms[mx_angle_idx], new_img_offset)
                                       
                        ## scales/angles
                        mx_scales[ilayer], mx_angles[ilayer] = scale_factors[mx_scale_idx], sparam_angles[mx_angle_idx]
                    ##
                    cur_rct[0:2] = np.mean(pos_wh[1:], axis =0) # 

                    #cur_contour = np.mean(rect_points, axis=0) # [4,2], [w,h]
                    
                    currentScaleFactor = currentScaleFactor * np.mean(mx_scales[0:6]) # ??
                    if currentScaleFactor < min_scale_factor: 
                        currentScaleFactor = min_scale_factor
                    if currentScaleFactor > max_scale_factor:
                        currentScaleFactor = max_scale_factor

                    currentAngleFactor = currentAngleFactor + np.mean(mx_angles)

                    ####
                    cur_rct[2:] = np.floor(target_sz*currentScaleFactor + 0.5) # ??
                    cur_rct = fun_border_modification(cur_rct,im.shape[0],im.shape[1])
                    pred_rcts[jj,:] = cur_rct
                    #pred_contours[jj,:] = cur_contour.flatten()

                    ####
                    str1 = "[%3d]:[%3.2f,%3.2f,%3.2f,%3.2f],[%3.2f,%3.2f,%3.2f,%3.2f], [%.3f], s[%.4f,%.4f]" %(jj,\
                                      gt_rcts[jj,0],gt_rcts[jj,1],gt_rcts[jj,2],gt_rcts[jj,3],\
                                      pred_rcts[jj,0], pred_rcts[jj,1],pred_rcts[jj,2],pred_rcts[jj,3],\
                                      update_factor, currentScaleFactor, currentAngleFactor)
                    #str2 = "\n\t\t gt_contours   [%s] \n\t\t pred_contours [%s]" %(\
                    #                    vector2string(gt_contours[jj,:],'float'), vector2string(pred_contours[jj,:],'float'))
                    str3 = "\n\t\t max_angles [%s]\n\t\t max_scales [%s] \n\t\t pos_wh [%s]" %(\
                                      vector2string(mx_angles,'float'),\
                                      vector2string(mx_scales,'float'),\
                                      vector2string(pos_wh.flatten(),'float'))
                                      #matrix2string(np.reshape(max_response,(-1, nn_p)),'float'))

                    logger.info(str1+str3)

                    str1 = '%04d.jpg' %(jj) #'T_%d.jpg'
                    fname = os.path.join(save_path,str1)
                    fun_draw_rct_on_image(X0[jj,:,:],fname, gt_rcts[jj,:],None, pred_rcts[jj,:])
                    #fun_draw_polygon_on_image(X0[jj,:,:],fname, gt_contours[jj,:], pred_contours[jj,:])

                    str1 = 'T_%d_mask.jpg' %(jj)
                    fname = os.path.join(save_path,str1)

                    str1 = 'prop_tr_%s_%d_%d.mat' %(fdr,iseq,jj)
                    fname = os.path.join(save_path,str1)

 
            #################################################################
            ########################### updating model ######################
            #################################################################
                ###########################
                #### << ctr + scale model >> ####
                ###########################
                # pdb.set_trace()
                old_tgt_ctr = np.floor(cur_rct[0:2] + 0.5) # [w,h]
                old_bbx_hw, old_bind, new_img_sz, new_img_offset, new_bbx_hw, new_bbx_hw_ratio = get_bounding_boxes_for_nscale(\
                                              old_tgt_ctr, scale_window_sz, currentScaleFactor, [1.0], padFactor=1.2) # [h,w]
                new_im = fun_get_patch(im, old_tgt_ctr, new_img_sz[::-1])
                # pdb.set_trace()
                #vgg.nangle = 1
                feed_dict = {vgg.imgs: np.expand_dims(new_im,0), \
                                 vgg.angle1: np.asarray([currentAngleFactor], dtype=np.float32), \
                                 vgg.img_sz: new_img_sz,     vgg.bbx: new_bbx_hw_ratio,         vgg.bind: old_bind, \
                                 vgg.crop_size: np.asarray([224,224], dtype=np.int32)}
                ## smp_patches: [al, sl, 224, 224, c]; smp_transforms: list of len=al, [8]; smp_offset: [2]; all them [h,w]
                smp_patches = vgg_sess.run(vgg.patches1, feed_dict=feed_dict)

                #smp_patches = np.reshape(smp_patches[0],(1,224,224,3))
                vgg_feas = extract_feature(smp_patches, cos_win = cos_win, vgg_out_norm_hw = fea_sz, vgg_ind = 2)

                # [1, fea_sz, nlayer*ch]                    
                #vgg_trfeas = np.reshape(vgg_trfeas,(1, fea_sz[1], fea_sz[0], nn_d))
                vgg_feas = np.reshape(vgg_feas[0],(nn_m,nn_d2))

                #### update model 
                if (jj % cf_nframe_update == 0): # or jj < 5): 
                    if jj==0:
                        feed_dict = {mdl_tr2.ph_data: vgg_feas, mdl_tr2.ph_labels: pmap}
                        ws = sess_tr2.run([mdl_tr2.ws],feed_dict=feed_dict)
                        ws = np.array(ws[0])

                        model_alpha2 = np.copy(ws)
                        #model_x =np.copy(vgg_trfea2)
                    else:
                        feed_dict = {mdl_tr2.ph_data: vgg_feas, mdl_tr2.ph_labels: pmap, mdl_tr2.ph_wt:model_alpha2}
                        ws = sess_tr2.run([mdl_tr2.ws_fast],feed_dict=feed_dict)
                        ws = np.array(ws[0])

                        if np.isnan(np.max(np.absolute(ws))) == False:
                            model_alpha2 = (1-update_factor)*model_alpha2 + update_factor*ws
                            #model_x     = (1-update_factor)*model_x    + update_factor*vgg_trfea2


                       #### ================================= 
	    ## save all results
            sess_tr.close()
            sess_te.close()
            sess_tr.close()
            sess_te2.close()
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
     
