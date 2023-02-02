import torch.nn as nn
from torchvision import transforms, utils
import os
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

def calculate_ious(gts, pseudos, n_classes=3):
    
    ious = []
    gts = gts.view(-1)
    pseudos = pseudos.view(-1)
    ignore_class_idx=0
    
    # for cls in range(1, n_classes+1):
    for cls in range(1, 9):
        pseudos_idxs = pseudos == cls
        gts_idxs = gts == cls
        intersection = (pseudos_idxs[gts_idxs]).long().sum().data.cpu()
        union = pseudos_idxs[gts != ignore_class_idx].long().sum().data.cpu() + \
                gts_idxs[gts != ignore_class_idx].long().sum().data.cpu() - \
                intersection

        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    print(ious)
    ious = np.array(ious)
    
    # log combined ious
    print_dict = {
        "mean IoU": np.nanmean(ious[:3]),
        "IoU car": ious[0],
        "IoU person": ious[1],
        "IoU bicycle": ious[2]
    }
    
    

    return ious, print_dict

def change_class_idx(pseudos,car,person,bicycle):
    
    pseudos[pseudos==1]=0
    for i in car: 
        pseudos[pseudos==i]=1
    
    pseudos[pseudos==2]=0
    for i in person: 
        pseudos[pseudos==i]=2
    
    pseudos[pseudos==3]=0
    for i in bicycle: 
        pseudos[pseudos==i]=3
    
    return pseudos

def visualization_eval(i,imgs, seg_preds,model_name):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans = transforms.ToPILImage()
    
    save_path = os.path.join('./log_files',model_name+'_checkpoints','test')
    
    if imgs.shape[1]==3:
        rgb = new_trans_rgb(imgs[0].cpu())
        rgb.save(os.path.join(save_path,'{}_img.png'.format(i)))
    else:
        th = new_trans(imgs[0].cpu())
        th.save(os.path.join(save_path,'{}_img.png'.format(i)))
    
    seg = seg_preds.squeeze(0).cpu().numpy()
    visualize_cityscapes_palette(os.path.join(save_path,'{}_seg.png'.format(i)), seg)
    
        
        


def eval_seg_RGB(testloader, model_name, NUM_TEST_DATASET, model):
    
    gts = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    preds = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    with torch.no_grad():  
        model.eval() 
        
        for i, batch in enumerate(testloader):
            rgb_imgs, labels = batch  # labels => #torch.Size([24(batch), 480, 640])
            
            batch_size = labels.shape[0]
            
            rgb_imgs = rgb_imgs.cuda()  # torch.Size([1, 1, 480, 640])
            # mid_pred1 = model.net_G_rgb(rgb_imgs) # torch.Size([1, 2048, 15, 20])
            # pred1 = model.decoder(mid_pred1).cpu() #torch.Size([1, 19, 480, 640])
            
            if 'MANet2' in model_name:
                mid_preds1, _ = model.net_G_rgb(rgb_imgs)
                pred1 = model.decoder(mid_preds1)
            else:
                mid_preds1 = model.net_G_rgb(rgb_imgs) # torch.Size([1, 2048, 15, 20])
                pred1,_ = model.decoder(mid_preds1) #torch.Size([1, 19, 480, 640])
            pred1 = pred1.cpu()
            seg_preds = torch.argmax(pred1,dim=1) #torch.Size([1, 480, 640])
            
            visualization_eval(i,rgb_imgs, seg_preds, model_name)
            
            for j in range(batch_size):
                gts[(i*batch_size)+j, :, :] = labels[j]
                
                preds[(i*batch_size)+j, :, :] = seg_preds[j]
                
    if (model_name == 'MS_UDA' or "default" in model_name) and 'gt' not in model_name:         
        preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    else:
        if 'pseudo' in model_name or 'attention' in model_name:
            preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    ious, print_dict = calculate_ious(gts, preds)
    
    return print_dict 

def eval_seg_MANet3_atloss(testloader, model_name, NUM_TEST_DATASET, model):
    
    gts = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    preds = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    with torch.no_grad():  
        model.eval() 
        
        for i, batch in enumerate(testloader):
            th_imgs, labels = batch  # labels => #torch.Size([24(batch), 480, 640])
            
            batch_size = labels.shape[0]
            
            th_imgs = th_imgs.cuda()  # torch.Size([1, 1, 480, 640])
        
            mid_preds2 = model.net_G_thermal(th_imgs) # torch.Size([1, 2048, 15, 20])
            mid_preds2_att, _ = model.attention_thermal(mid_preds2)
            pred2 = model.decoder(mid_preds2_att).cpu() #torch.Size([1, 19, 480, 640])
            seg_preds = torch.argmax(pred2,dim=1) #torch.Size([1, 480, 640])
            
            visualization_eval(i,th_imgs, seg_preds, model_name)
            
            for j in range(batch_size):
                gts[(i*batch_size)+j, :, :] = labels[j]
                
                preds[(i*batch_size)+j, :, :] = seg_preds[j]
                
    if (model_name == 'MS_UDA' or "default" in model_name) and 'gt' not in model_name:         
        preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    else:
        if 'pseudo' in model_name or 'attention' in model_name:
            preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    ious, print_dict = calculate_ious(gts, preds)
    
    return print_dict


def eval_seg_MANet3(testloader, model_name, NUM_TEST_DATASET, model):
    
    gts = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    preds = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    with torch.no_grad():  
        model.eval() 
        
        for i, batch in enumerate(testloader):
            th_imgs, labels = batch  # labels => #torch.Size([24(batch), 480, 640])
            
            batch_size = labels.shape[0]
            
            th_imgs = th_imgs.cuda()  # torch.Size([1, 1, 480, 640])
        
            mid_preds2 = model.net_G_thermal(th_imgs) # torch.Size([1, 2048, 15, 20])
            mid_preds2_att = model.attention_thermal(mid_preds2)
            pred2 = model.decoder(mid_preds2_att).cpu() #torch.Size([1, 19, 480, 640])
            seg_preds = torch.argmax(pred2,dim=1) #torch.Size([1, 480, 640])
            
            visualization_eval(i,th_imgs, seg_preds, model_name)
            
            for j in range(batch_size):
                gts[(i*batch_size)+j, :, :] = labels[j]
                
                preds[(i*batch_size)+j, :, :] = seg_preds[j]
                
    if (model_name == 'MS_UDA' or "default" in model_name) and 'gt' not in model_name:         
        preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    else:
        if 'pseudo' in model_name or 'attention' in model_name:
            preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    ious, print_dict = calculate_ious(gts, preds)
    
    return print_dict

def eval_seg_atloss(testloader, model_name, NUM_TEST_DATASET, model):
    
    gts = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    preds = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    with torch.no_grad():  
        model.eval() 
        
        for i, batch in enumerate(testloader):
            th_imgs, labels = batch  # labels => #torch.Size([24(batch), 480, 640])
            
            batch_size = labels.shape[0]
            
            th_imgs = th_imgs.cuda()  # torch.Size([1, 1, 480, 640])
            
            if 'MANet2' in model_name:
                mid_preds2, _ = model.net_G_thermal(th_imgs)
                pred2 = model.decoder(mid_preds2)
            else:
                mid_pred2 = model.net_G_thermal(th_imgs) # torch.Size([1, 2048, 15, 20])
                pred2,_ = model.decoder(mid_pred2) #torch.Size([1, 19, 480, 640])
            pred2 = pred2.cpu()
            seg_preds = torch.argmax(pred2,dim=1) #torch.Size([1, 480, 640])
            
            visualization_eval(i,th_imgs, seg_preds, model_name)
            
            for j in range(batch_size):
                gts[(i*batch_size)+j, :, :] = labels[j]
                
                preds[(i*batch_size)+j, :, :] = seg_preds[j]
                
    if (model_name == 'MS_UDA' or "default" in model_name) and 'gt' not in model_name:         
        preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    else:
        if 'pseudo' in model_name or 'attention' in model_name:
            preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    ious, print_dict = calculate_ious(gts, preds)
    
    return print_dict  


def eval_seg(testloader, model_name, NUM_TEST_DATASET, model):
    
    gts = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    preds = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    with torch.no_grad():  
        model.eval() 
        
        for i, batch in enumerate(testloader):
            th_imgs, labels = batch  # labels => #torch.Size([24(batch), 480, 640])
            
            batch_size = labels.shape[0]
            
            th_imgs = th_imgs.cuda()  # torch.Size([1, 1, 480, 640])
            mid_pred2 = model.net_G_thermal(th_imgs) # torch.Size([1, 2048, 15, 20])
            pred2 = model.decoder(mid_pred2).cpu() #torch.Size([1, 19, 480, 640])
            seg_preds = torch.argmax(pred2,dim=1) #torch.Size([1, 480, 640])
            
            visualization_eval(i,th_imgs, seg_preds, model_name)
            
            for j in range(batch_size):
                gts[(i*batch_size)+j, :, :] = labels[j]
                
                preds[(i*batch_size)+j, :, :] = seg_preds[j]
                
    if (model_name == 'MS_UDA' or "default" in model_name) and 'gt' not in model_name:         
        preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    else:
        if 'pseudo' in model_name or 'attention' in model_name:
            preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    ious, print_dict = calculate_ious(gts, preds)
    
    return print_dict     
            
def eval_seg_edge(testloader, model_name, NUM_TEST_DATASET, model):
    
    gts = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    preds = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    with torch.no_grad():  
        model.eval() 
        
        for i, batch in enumerate(testloader):
            th_imgs, labels = batch  # labels => #torch.Size([24(batch), 480, 640])
            
            batch_size = labels.shape[0]
            
            th_imgs = th_imgs.cuda()  # torch.Size([1, 1, 480, 640])
            _,_,h,w = th_imgs.shape
            mid_pred2 = model.net_G_thermal(th_imgs)   #출력 : [x1,x2,x3,x4]
            out2, edge2 = model.decoder(mid_pred2,h,w)     #출력 : x_fuse, edge
            out2.append(edge2)
            class_maps2 = model.last_layer(torch.cat(out2, 1)).cpu()  #최종적으로 class_maps , edge 를 사용
            seg_preds = torch.argmax(class_maps2,dim=1)

            
            for j in range(batch_size):
                gts[(i*batch_size)+j, :, :] = labels[j]
                
                preds[(i*batch_size)+j, :, :] = seg_preds[j]
                
    if (model_name == 'MS_UDA' or "default" in model_name) and 'gt' not in model_name:         
        preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    else:
        if 'pseudo' in model_name:
            preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    ious, print_dict = calculate_ious(gts, preds)
    
    return print_dict     

def eval_seg_edge_dec(testloader, model_name, NUM_TEST_DATASET, model):
    
    gts = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    preds = Variable(torch.zeros(NUM_TEST_DATASET, 480, 640)).cuda()
    with torch.no_grad():  
        model.eval() 
        
        for i, batch in enumerate(testloader):
            th_imgs, labels = batch  # labels => #torch.Size([24(batch), 480, 640])
            
            batch_size = labels.shape[0]
            
            th_imgs = th_imgs.cuda()  # torch.Size([1, 1, 480, 640])
            _,_,h,w = th_imgs.shape
            mid_pred2 = model.net_G_thermal(th_imgs)   #출력 : [x1,x2,x3,x4]
            out2 = model.decoder(mid_pred2,h,w)     #출력 : x_fuse, edge
            class_maps2 = model.last_layer(torch.cat(out2, 1)).cpu()  #최종적으로 class_maps , edge 를 사용
            seg_preds = torch.argmax(class_maps2,dim=1)

            
            for j in range(batch_size):
                gts[(i*batch_size)+j, :, :] = labels[j]
                
                preds[(i*batch_size)+j, :, :] = seg_preds[j]
                
    if (model_name == 'MS_UDA' or "default" in model_name) and 'gt' not in model_name:         
        preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    else:
        if 'pseudo' in model_name:
            preds = change_class_idx(preds,[13,14,15],[11,12],[18,17])
    ious, print_dict = calculate_ious(gts, preds)
    
    return print_dict    


def visualize_edge(edges, model_name, rgb_images, th_images, gts, pseudo_labels, pred_rgb, pred_th, names, check_dir, epoch=None, val=False):   
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])

    new_trans = transforms.ToPILImage()

    if val is False:
        show_dir = os.path.join(check_dir, 'show')
    else:
        show_dir = os.path.join(check_dir, 'val_show')
    os.makedirs(show_dir, exist_ok=True)

    for i in range(len(names)):
        if epoch is not None:
            edge_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_edge.png')
            rgb_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            gt_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_gt.png')
            pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th_pred.png')
        else:
            edge_path = os.path.join(show_dir, names[i] + '_edge.png')
            rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            gt_path = os.path.join(show_dir, names[i] + '_gt.png')
            pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, names[i] + '_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, names[i] + '_th_pred.png')

        rgb_img = rgb_images[i]
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)

        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)
        
        # edge_img =(torch.sigmoid(edges[i])>=0.5)*255
        edge_img = (edges[i]>0)*255
        edge_img = edge_img.permute(1,2,0)
        img = new_trans(np.uint8(edge_img.cpu().numpy()))
        img.save(edge_path)
    ################
    
        pseudo_img = pseudo_labels[i].cpu().numpy()
        visualize_cityscapes_palette(pseudo_path, pseudo_img)

        gt_img = gts[i].cpu().numpy()
        visualize_mfnet_palette(gt_path, gt_img)

        rgb_pred_img = pred_rgb[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(rgb_pred_path, rgb_pred_img)

        th_pred_img = pred_th[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(th_pred_path, th_pred_img) 
        
def visualize_edge_supervised(edges, model_name, rgb_images, th_images, gts, pseudo_labels, pred_rgb, pred_th, names, check_dir, epoch=None, val=False):   
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])

    new_trans = transforms.ToPILImage()

    if val is False:
        show_dir = os.path.join(check_dir, 'show')
    else:
        show_dir = os.path.join(check_dir, 'val_show')
    os.makedirs(show_dir, exist_ok=True)

    for i in range(len(names)):
        if epoch is not None:
            edge_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_edge.png')
            rgb_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            gt_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_gt.png')
            pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th_pred.png')
        else:
            edge_path = os.path.join(show_dir, names[i] + '_edge.png')
            rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            gt_path = os.path.join(show_dir, names[i] + '_gt.png')
            pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, names[i] + '_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, names[i] + '_th_pred.png')

        rgb_img = rgb_images[i]
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)

        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)
        
        # edge_img =(torch.sigmoid(edges[i])>=0.5)*255
        edge_img = (edges[i]>0)*255
        edge_img = edge_img.permute(1,2,0)
        img = new_trans(np.uint8(edge_img.cpu().numpy()))
        img.save(edge_path)
    ################
    
        gt_img = gts[i].cpu().numpy()
        visualize_mfnet_palette(gt_path, gt_img)

        rgb_pred_img = pred_rgb[i].argmax(0).cpu().numpy()
        visualize_mfnet_palette(rgb_pred_path, rgb_pred_img)

        th_pred_img = pred_th[i].argmax(0).cpu().numpy()
        visualize_mfnet_palette(th_pred_path, th_pred_img) 
        



def visualize(model_name, rgb_images, th_images, gts, pseudo_labels, pred_rgb, pred_th, names, check_dir, epoch=None, val=False):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])

    new_trans = transforms.ToPILImage()

    if val is False:
        show_dir = os.path.join(check_dir, 'show')
    else:
        show_dir = os.path.join(check_dir, 'val_show')
    os.makedirs(show_dir, exist_ok=True)

    for i in range(len(names)):
        if epoch is not None:
            rgb_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            gt_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_gt.png')
            pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th_pred.png')
        else:
            rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            gt_path = os.path.join(show_dir, names[i] + '_gt.png')
            pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, names[i] + '_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, names[i] + '_th_pred.png')

        rgb_img = rgb_images[i]
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)

        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)


        pseudo_img = pseudo_labels[i].cpu().numpy()
        visualize_cityscapes_palette(pseudo_path, pseudo_img)

        # if 'gt' in model_name:    
        #     gt_img = gts[i].cpu().numpy()
        #     visualize_mfnet_palette(gt_path, gt_img)

        #     rgb_pred_img = pred_rgb[i].argmax(0).cpu().numpy()
        #     visualize_mfnet_palette(rgb_pred_path, rgb_pred_img)

        #     th_pred_img = pred_th[i].argmax(0).cpu().numpy()
        #     visualize_mfnet_palette(th_pred_path, th_pred_img)    
        
        # else:
        gt_img = gts[i].cpu().numpy()
        visualize_mfnet_palette(gt_path, gt_img)

        rgb_pred_img = pred_rgb[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(rgb_pred_path, rgb_pred_img)

        th_pred_img = pred_th[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(th_pred_path, th_pred_img) 

####################################################################################################################

def visualize_supervised(model_name, th_images, gts, pred2, names, check_dir, epoch=None, val=False):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])

    new_trans = transforms.ToPILImage()

    if val is False:
        show_dir = os.path.join(check_dir, 'show')
    else:
        show_dir = os.path.join(check_dir, 'val_show')
    os.makedirs(show_dir, exist_ok=True)

    for i in range(len(names)):
        if epoch is not None:
            # rgb_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            gt_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_gt.png')
            # pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            # rgb_pred_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th_pred.png')
        else:
            # rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            gt_path = os.path.join(show_dir, names[i] + '_gt.png')
            # pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            # rgb_pred_path = os.path.join(show_dir, names[i] + '_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, names[i] + '_th_pred.png')

        # rgb_img = rgb_images[i]
        # rgb = new_trans_rgb(rgb_img.cpu())
        # rgb.save(rgb_path)

        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)
           
        gt_img = gts[i].cpu().numpy()
        visualize_mfnet_palette(gt_path, gt_img)

        th_pred_img = pred2[i].argmax(0).cpu().numpy()
        visualize_mfnet_palette(th_pred_path, th_pred_img)    
        
        
def visualize_pseudo(model_name, th_images, gts, pred2, names, check_dir, epoch=None, val=False):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])

    new_trans = transforms.ToPILImage()

    if val is False:
        show_dir = os.path.join(check_dir, 'show')
    else:
        show_dir = os.path.join(check_dir, 'val_show')
    os.makedirs(show_dir, exist_ok=True)

    for i in range(len(names)):
        if epoch is not None:
            # rgb_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            gt_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_gt.png')
            # pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            # rgb_pred_path = os.path.join(show_dir, str(epoch)+'_'+names[i]+'_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th_pred.png')
        else:
            # rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            gt_path = os.path.join(show_dir, names[i] + '_gt.png')
            # pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            # rgb_pred_path = os.path.join(show_dir, names[i] + '_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, names[i] + '_th_pred.png')

        # rgb_img = rgb_images[i]
        # rgb = new_trans_rgb(rgb_img.cpu())
        # rgb.save(rgb_path)

        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)
           
        gt_img = gts[i].cpu().numpy()
        visualize_mfnet_palette(gt_path, gt_img)

        th_pred_img = pred2[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(th_pred_path, th_pred_img)  
 #######################################################################################      
        

def lr_with_epo(optimizer, base_lr, epo, lr_decay=0.95, print_set=False):

    lr = base_lr * lr_decay ** (epo - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if print_set:
        return lr


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, max_iter, base_lr, power, print_set=False):
    # update per iter
    lr = lr_poly(base_lr, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr

    if print_set:
        return lr


def find_norm(norm_name):
    if norm_name.count('InstanceNorm'):
        norm = nn.InstanceNorm2d
    elif norm_name.count('BatchNorm'):
        norm = nn.BatchNorm2d

    return norm


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def visualizer(rgb_images, th_images, pred1, pred2, labels, names, results_dir):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans_th = transforms.ToPILImage()

    for i in range(len(names)):
        rgb_img = rgb_images[i]
        rgb_path = os.path.join(results_dir, names[i] + '_rgb.png')
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)

        th_img = th_images[i]
        th_path = os.path.join(results_dir, names[i] + '_th.png')
        th = new_trans_th(th_img.cpu())
        th.save(th_path)

        label = labels[i]
        label_path = os.path.join(results_dir, names[i] + '_label.png')
        visualize_single(label_path, label)

        pred1_path = os.path.join(results_dir, names[i] + '_rgb_pred.png')
        rgb_pred = pred1[i]
        rgb_pred = rgb_pred.argmax(0)
        visualize_single(pred1_path, rgb_pred)

        pred2_path = os.path.join(results_dir, names[i] + '_th_pred.png')
        th_pred = pred2[i]
        th_pred = th_pred.argmax(0)
        visualize_single(pred2_path, th_pred)


def visualizer_one_sensor(images, preds, check_type, labels, names, results_dir, epoch=None):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    if check_type in ['RGB', 'RGB_N']:
        new_trans = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    else:
        new_trans = transforms.ToPILImage()

    for i in range(len(names)):
        img = images[i]
        if epoch is not None:
            img_path = os.path.join(results_dir, str(epoch) + '_' + names[i] + '.png')
        else:
            img_path = os.path.join(results_dir, names[i] + '.png')
        # img_path = os.path.join(results_dir, names[i] + '.png')
        img = new_trans(img.cpu())
        img.save(img_path)

        label = labels[i]
        if epoch is not None:
            label_path = os.path.join(results_dir, str(epoch) + '_' + names[i] + '_label.png')
        else:
            label_path = os.path.join(results_dir, names[i] + '_label.png')
        visualize_single(label_path, label)

        pred = preds[i]
        if epoch is not None:
            pred_path = os.path.join(results_dir, str(epoch) + '_' + names[i] + '_pred.png')
        else:
            pred_path = os.path.join(results_dir, names[i] + '_label.png')
        pred = pred.argmax(0)
        visualize_single(pred_path, pred)


def visualizer_single(rgb_image, th_image, pred1, pred2, label, check_dir, total_iters, name):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans_th = transforms.ToPILImage()

    show_folder = os.path.join(check_dir, 'show')
    if not os.path.exists(show_folder):
        os.makedirs(show_folder)

    rgb_path = os.path.join(show_folder, str(total_iters) + '_iters_rgb_' + name + '.png')
    rgb = new_trans_rgb(rgb_image.cpu())
    rgb.save(rgb_path)

    th_path = os.path.join(show_folder, str(total_iters) + '_iters_th_' + name + '.png')
    th = new_trans_th(th_image.cpu())
    th.save(th_path)

    label_path = os.path.join(show_folder, str(total_iters) + '_iters_label_' + name + '.png')
    visualize_single(label_path, label)

    rgb_pred_path = os.path.join(show_folder, str(total_iters) + '_iters_rgb_pred_' + name + '.png')
    rgb_pred = pred1.argmax(0)
    visualize_single(rgb_pred_path, rgb_pred)

    th_pred_path = os.path.join(show_folder, str(total_iters) + '_iters_th_pred_' + name + '.png')
    th_pred = pred2.argmax(0)
    visualize_single(th_pred_path, th_pred)


def cross_rgb_trans(rgb_images):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    b, c, h, w = rgb_images.size()
    new_gray = torch.zeros((b, 1, h, w))
    for i in range(rgb_images.size(0)):
        tmp = invTrans(rgb_images[i])
        new_gray[i] = rgb_to_gray_single(tmp)
    return new_gray


def rgb_to_gray_single(tmp_rgb):
    tmp_gray = 0.299 * tmp_rgb[0] + 0.587 * tmp_rgb[1] + 0.114 * tmp_rgb[2]
    return tmp_gray


def get_palette():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128] #[0, 0, 142]#
    person = [64, 64, 0] #[220, 20, 60] #
    bike =[0, 128, 192] #[119, 11, 32] #[
    curve = [0, 0, 192] #[0,0,0]
    car_stop = [128, 128, 0] #[0,0,0]
    guardrail = [64, 64, 128] #[0,0,0]
    color_cone = [192, 128, 128] #[0,0,0]
    bump = [192, 64, 0] #[0,0,0] #

    palette = np.array([unlabelled, car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize_single(image_name, prediction):
    palette = get_palette()

    """
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(image_name)
    """

    pred = prediction.cpu().numpy()
    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(0, len(palette)):  # fix the mistake from the MFNet code on Dec.27, 2019
        img[pred == cid] = palette[cid]

    img = Image.fromarray(np.uint8(img))
    img.save(image_name)


def save_pseudo_label(preds, names, results_dir):
    for i in range(len(names)):
        pred_path = os.path.join(results_dir, names[i] + '_pseudo.png')
        pred = preds[i]
        pred = pred.argmax(0)
        prediction = pred.cpu().numpy()
        img = Image.fromarray(np.uint8(prediction))
        img.save(pred_path)


def TL_visualize(rgb_images, predictions, names, show_dir):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])

    for i in range(len(names)):
        rgb_img = rgb_images[i]
        rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)

        pseudo_img = predictions[i].cpu().numpy()
        pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
        img = Image.fromarray(np.uint8(pseudo_img))
        img.save(pseudo_path)

        pred_img = predictions[i].cpu().numpy()
        pred_path = os.path.join(show_dir, names[i] + '_pred.png')
        visualize_cityscapes_palette(pred_path, pred_img)

def TL_rgb_th_visualize(rgb_images, th_images, pseudo_labels, predictions, names, check_dir, epoch=None):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                  transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                       std=[1., 1., 1.]),
                                  ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans = transforms.ToPILImage()

    show_dir = os.path.join(check_dir, 'show')
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    for i in range(len(names)):

        if epoch is not None:
            rgb_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pred.png')
        else:
            rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            pred_path = os.path.join(show_dir, names[i] + '_pred.png')

        rgb_img = rgb_images[i]
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)
        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)
        pseudo_img = pseudo_labels[i].cpu().numpy()
        visualize_cityscapes_palette(pseudo_path, pseudo_img)
        pred_img = predictions[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(pred_path, pred_img)
####################################################################################################################
def visualize_MF(rgb_images, th_images, labels, preds, names, check_dir, epoch =None):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                  transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                       std=[1., 1., 1.]),
                                  ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans = transforms.ToPILImage()

    show_dir = os.path.join(check_dir, 'show')
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    for i in range(len(names)):

        if epoch is not None:
            rgb_path = os.path.join(show_dir, str(epoch)+'_'+names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, str(epoch)+'_'+names[i] + '_th.png')
            pred_path = os.path.join(show_dir, str(epoch)+'_'+names[i] + '_pred.png')
            label_path = os.path.join(show_dir, str(epoch)+'_'+names[i] + '_label.png')
        else:
            rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            pred_path = os.path.join(show_dir, names[i] + '_pred.png')
            label_path = os.path.join(show_dir, names[i] + '_label.png')

        if rgb_images is not None:
            rgb_img = rgb_images[i]
            rgb = new_trans_rgb(rgb_img.cpu())
            rgb.save(rgb_path)
        if th_images is not None:
            th_img = th_images[i]
            th = new_trans(th_img.cpu())
            th.save(th_path)
        if preds is not None:
            pred_img = preds[i]
            pred_img = pred_img.argmax(0)
            visualize_single(pred_path, pred_img)
        label_img = labels[i]
        visualize_single(label_path, label_img)
############################################################################################################

def visualize_cityscapes_palette(image_name, pred):
    cityscapes_palette = fill_colormap()

    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(0, len(cityscapes_palette)):  # fix the mistake from the MFNet code on Dec.27, 2019
        img[pred == cid] = cityscapes_palette[cid]
    img = Image.fromarray(np.uint8(img))
    img.save(image_name)

#[13,14,15],[11,12],[18,17]
def fill_colormap():
    road = [128, 64, 128] #0
    sidewalk = [244, 35, 232]
    building = [70, 70, 70]
    wall = [102, 102, 156]
    fence = [190, 153, 153]
    pole = [153, 153, 153]
    traffic_light = [250, 170, 30]
    traffic_sign = [220, 220, 0]
    vegetation = [107, 142, 35]
    terrain = [152, 251, 152]
    sky = [70, 130, 180]
    person = [220, 20, 60]
    rider = [255, 0, 0]
    car = [0, 0, 142]
    truck = [0, 0, 70]
    bus = [0, 60, 100]
    train = [0, 80, 100]
    motorcycle = [0, 0, 230]
    bicycle = [119, 11, 32] #18

    palette = np.array([road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign,
                        vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle])

    return palette
###########################################################################################

def visualize_mfnet_palette(image_name, pred):
    mfnet_palette = fill_colormap_mfnet()

    img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(0, len(mfnet_palette)):  # fix the mistake from the MFNet code on Dec.27, 2019
        img[pred == cid] = mfnet_palette[cid]
    img = Image.fromarray(np.uint8(img))
    img.save(image_name)


def fill_colormap_mfnet():
    unlabeled = [128, 64, 128]
    car = [0, 0, 142]
    person = [220, 20, 60]
    bicycle = [119, 11, 32]
    curve = [244, 35, 232]
    car_stop = [70, 70, 70]
    guardrail = [102, 102, 156]
    color_cone = [190, 153, 153]
    bump = [153, 153, 153]

    palette = np.array([unlabeled, car, person, bicycle, curve, car_stop, guardrail, color_cone, bump])

    return palette

def kl_weight_func(epoch, max_epochs):
    
    div = float(epoch)/max_epochs
    
    if div<0.2:
        return 0.2
    else:
        return div
    



#################################################################################################################
def load_state_from_model(pretrained_weight, model, gpus):
    own_state = model.state_dict()
    # pretraiend_weight: part.module.layer2....
    # model.state_dict(): part.layer2...(single gpu)

    for name, param in pretrained_weight.items():
        if len(gpus) == 1:
            new_name = name.split('.module')
            new_name = "".join(new_name)
        else:
            new_name = name
        if new_name not in own_state:
            continue
        own_state[new_name].copy_(param)
    print('done!')

# FOR KP Evaluation
####################################################################################################################################
def mono_sup_visualize_KP(rgb_images, th_images, labels, pseudo_labels, predictions, names, check_dir,
                                epoch=None):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans = transforms.ToPILImage()

    show_dir = os.path.join(check_dir, 'show')
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    for i in range(len(names)):

        if epoch is not None:
            rgb_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pred.png')
            label_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_label.png')
        else:
            rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            pred_path = os.path.join(show_dir, names[i] + '_pred.png')
            label_path = os.path.join(show_dir, names[i] + '_label.png')

        if rgb_images is not None:
            rgb_img = rgb_images[i]
            rgb = new_trans_rgb(rgb_img.cpu())
            rgb.save(rgb_path)
        if th_images is not None:
            th_img = th_images[i]
            th = new_trans(th_img.cpu())
            th.save(th_path)
        if pseudo_labels is not None:
            pseudo_img = pseudo_labels[i].cpu().numpy()
            visualize_cityscapes_palette(pseudo_path, pseudo_img)
        if predictions is not None:
            pred_img = predictions[i].argmax(0).cpu().numpy()
            visualize_cityscapes_palette(pred_path, pred_img)
        if labels is not None:
            label = labels[i]
            label_img = label.cpu().numpy()
            visualize_cityscapes_palette_GT(label_path, label_img)

# KP
def TL_DA_visualize_with_GT(rgb_images, th_images, labels, pseudo_labels, pred1, pred2, names, check_dir, epoch=None):

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans = transforms.ToPILImage()

    show_dir = os.path.join(check_dir, 'show')
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    for i in range(len(names)):

        if epoch is not None:
            rgb_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th.png')
            pseudo_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_th_pred.png')
            label_path = os.path.join(show_dir, str(epoch) + '_' + names[i] + '_label.png')
        else:
            rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
            th_path = os.path.join(show_dir, names[i] + '_th.png')
            pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
            rgb_pred_path = os.path.join(show_dir, names[i] + '_rgb_pred.png')
            th_pred_path = os.path.join(show_dir, names[i] + '_th_pred.png')
            label_path = os.path.join(show_dir, names[i] + '_label.png')

        rgb_img = rgb_images[i]
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)
        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)
        pseudo_img = pseudo_labels[i].cpu().numpy()
        visualize_cityscapes_palette(pseudo_path, pseudo_img)
        rgb_pred_img = pred1[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(rgb_pred_path, rgb_pred_img)
        th_pred_img = pred2[i].argmax(0).cpu().numpy()
        visualize_cityscapes_palette(th_pred_path, th_pred_img)
        label = labels[i]
        label_img = label.cpu().numpy()
        visualize_cityscapes_palette_GT(label_path, label_img)

def TL_visualize_with_GT(rgb_images, th_images, labels, pseudo_labels, names, check_dir):
    #for KP

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])
    new_trans_rgb = transforms.Compose([invTrans,
                                        transforms.ToPILImage()])
    new_trans = transforms.ToPILImage()

    show_dir = os.path.join(check_dir, 'show')
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    for i in range(len(names)):


        rgb_path = os.path.join(show_dir, names[i] + '_rgb.png')
        th_path = os.path.join(show_dir, names[i] + '_th.png')
        pseudo_path = os.path.join(show_dir, names[i] + '_pseudo.png')
        label_path = os.path.join(show_dir, names[i] + '_label.png')

        rgb_img = rgb_images[i]
        rgb = new_trans_rgb(rgb_img.cpu())
        rgb.save(rgb_path)
        th_img = th_images[i]
        th = new_trans(th_img.cpu())
        th.save(th_path)
        pseudo_img = pseudo_labels[i].cpu().numpy()
        visualize_cityscapes_palette(pseudo_path, pseudo_img)
        label = labels[i]
        label_img = label.cpu().numpy()
        visualize_cityscapes_palette_GT(label_path, label_img)

def fill_colormap_GT():
    road = [128, 64, 128]
    sidewalk = [244, 35, 232]
    building = [70, 70, 70]
    wall = [102, 102, 156]
    fence = [190, 153, 153]
    pole = [153, 153, 153]
    traffic_light = [250, 170, 30]
    traffic_sign = [220, 220, 0]
    vegetation = [107, 142, 35]
    terrain = [152, 251, 152]
    sky = [70, 130, 180]
    person = [220, 20, 60]
    rider = [255, 0, 0]
    car = [0, 0, 142]
    truck = [0, 0, 70]
    bus = [0, 60, 100]
    train = [0, 80, 100]
    motorcycle = [0, 0, 230]
    bicycle = [119, 11, 32]

    void = [0, 0, 0]

    palette = np.array([road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign,
                        vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle, void])

    return palette


def visualize_cityscapes_palette_GT(image_name, label):
    cityscapes_palette = fill_colormap_GT()

    img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cid in range(0, len(cityscapes_palette)):  # fix the mistake from the MFNet code on Dec.27, 2019
        img[label == cid] = cityscapes_palette[cid]
    img = Image.fromarray(np.uint8(img))
    img.save(image_name)


#######################################################################################################################