import os
import sys
import datetime

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from utils import save_img
from losses import CharbonnierLoss, DINOLoss, SeamLoss

import kornia
from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import wandb

from utils.loader import get_training_data, get_validation_data
from utils.image_utils import convert_color_space, imsave, rgb_to_hsv


######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



######### Model ###########
model_restoration = utils.get_arch(opt)

# # parameter count
# params = 0
# for p in model_restoration.parameters():
#     if p.requires_grad:
#         params += p.numel()
        
# print(params)  # 55484461

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration,path_chk_rest, opt=opt)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    opt.warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)
elif opt.pretrain_weights:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration,path_chk_rest, opt=opt)

# ######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### W & B ###########
if opt.wo_wandb:
    mode = "disabled"
else:
    mode = "online"
wandb.init(project="NTIRE2023_ShadowRemoval_IIM_TTI", config=opt, name=opt.env[1:], mode=mode)
wandb.watch(model_restoration)

######### Loss ###########
criterion = CharbonnierLoss(m_diff_alpha=opt.m_diff_alpha, m_shadow_alpha=opt.m_shadow_alpha, color_space=opt.color_space).cuda()
dino = DINOLoss().cuda()
seam_loss = SeamLoss(**opt.seam_condition).cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train, color_space=opt.color_space, mask_dir=opt.mask_dir, opt=opt)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir, color_space=opt.color_space, mask_dir=opt.mask_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = 1000
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
ii=0
index = 0
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = {'crite': 0, 'dino': 0, 'seam': 0, 'self_rep': 0, 'ft':0, 'shadow': 0, 'sum': 0}
    psnr_list = []
    train_id = 1
    epoch_ssim_loss = 0
    pbar = tqdm(train_loader, ncols=100)
    for i, data in enumerate(pbar, 0): 
        epoch_loss_formatted = dict()
        for key, value in epoch_loss.items():
            if key == 'dino' and not opt.dino_lambda: continue
            if key == 'seam' and not opt.seam_lambda: continue
            if key == 'self_rep' and not opt.self_rep_lambda: continue
            if key == 'ft' and not opt.self_feature_lambda: continue
            if key == 'shadow' and not opt.joint_learning_alpha: continue
            if key in ['dino', 'ft']:
                epoch_loss_formatted[key] = f"{value:.1e}"
            else:
                epoch_loss_formatted[key] = f"{value:.2f}"
        pbar.set_postfix(epoch_loss_formatted)
        # zero_grad
        loss = 0.0
        index += 1
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()
        if 'official_warped' in opt.train_dir:
            diff = data[3].cuda()
        else:
            diff = 0
        mask_edge = None
        if opt.joint_learning_alpha:
            mask_number_per = data[4].cuda()
            canny = kornia.filters.Canny()
            _, mask_edge = canny(mask)
            mask_edge = mask_edge.cuda()
        if (epoch > 5) and (not opt.nomixup):
            target, input_, mask, mask_edge = utils.MixUp_AUG().aug(target, input_, mask, mask_edge)
        if opt.w_hsv:
            hsv = rgb_to_hsv(input_)
            input_ = torch.cat((input_, hsv), dim=1)

        # self-representation learning 
        if opt.self_rep_lambda and not opt.self_rep_once:
            loss_self = 0.0
            with torch.cuda.amp.autocast():
                target_mask = torch.zeros_like(mask).cuda()
                restored, feature_target = model_restoration(target, target_mask)
                restored = torch.clamp(restored,0,1)
                if opt.color_space == 'hsv':
                    loss_self = loss_self + criterion(restored[:, 2], target[:, 2])
                else:
                    loss_self = loss_self + criterion(restored, target, diff)
                
                if opt.dino_lambda:
                    loss_dino = dino(restored, target)
                    loss_self = loss_self + opt.dino_lambda * loss_dino
                if opt.seam_lambda:
                    loss_seam = seam_loss(restored, target)
                    loss_self = loss_self + opt.seam_lambda * loss_seam

            epoch_loss['self_rep'] += loss_self.item()
            loss_self = loss_self * opt.self_rep_lambda
            epoch_loss['sum'] += loss_self.item()
            loss_scaler(
                    loss_self, optimizer,parameters=model_restoration.parameters())

        with torch.cuda.amp.autocast():
            if opt.self_rep_lambda and opt.self_rep_once:
                loss_self = 0.0
                target_mask = torch.zeros_like(mask).cuda()
                restored, feature_target = model_restoration(target, target_mask)
                restored = torch.clamp(restored,0,1)
                if opt.color_space == 'hsv':
                    loss_self = loss_self + criterion(restored[:, 2], target[:, 2])
                else:
                    loss_self = loss_self + criterion(restored, target, diff)
                if opt.dino_lambda:
                    loss_dino = dino(restored, target)
                    loss_self = loss_self + opt.dino_lambda * loss_dino
                if opt.seam_lambda:
                    loss_seam = seam_loss(restored, target)
                    loss_self = loss_self + opt.seam_lambda * loss_seam
                epoch_loss['self_rep'] += loss_self.item()
                loss = loss + loss_self * opt.self_rep_lambda

            if opt.joint_learning_alpha:
                restored, restored_mask, loss_shadow, feature_input = model_restoration(input_, mask, mask_edge, mask_number_per)
                loss_shadow = torch.sum(loss_shadow)
            else:
                restored, feature_input = model_restoration(input_, mask)
            restored = torch.clamp(restored,0,1)
            if opt.color_space == 'hsv':
                loss_cr = criterion(restored[:, 2], target[:, 2])
            else:
                loss_cr = criterion(restored, target, diff)
            loss = loss + loss_cr
            epoch_loss['crite'] += loss_cr.item()

            if opt.dino_lambda:
                loss_dino = dino(restored, target)
                loss = loss + opt.dino_lambda * loss_dino
                epoch_loss['dino'] += loss_dino.item()
            if opt.seam_lambda:
                loss_seam = seam_loss(restored, target)
                loss = loss + opt.seam_lambda * loss_seam
                epoch_loss['seam'] += loss_seam.item()

            if opt.self_feature_lambda:
                # print(feature_target.shape, feature_input.shape)
                loss_feature = F.mse_loss(feature_target, feature_input)
                loss = loss + opt.self_feature_lambda * loss_feature
                epoch_loss['ft'] += loss_feature.item()
            
            if opt.joint_learning_alpha:
                loss = (1 - opt.joint_learning_alpha) * loss + opt.joint_learning_alpha * loss_shadow
                epoch_loss['shadow'] += loss_shadow.item()
            
            epoch_loss['sum'] += loss.item()

            psnr_list.append(utils.batch_PSNR(restored, target, False, color_space=opt.color_space).item())
            filenames = data[4]
            if opt.joint_learning_alpha:
                filenames = data[5]
            if epoch in map(lambda x: ((x - 1) // len(train_loader)) + 1, range(eval_now, len(train_loader) * 1001, eval_now)) and i>0:
                for j, filename in enumerate(filenames):
                    if filename in ['0004.png', '0891.png', '0360.png', '0392.png']:
                        result_dir = os.path.join(log_dir, 'results', str(epoch))
                        os.makedirs(result_dir, exist_ok=True)
                        psnr = utils.myPSNR(restored[j], target[j]).item()
                        restored_save = restored[j].detach().cpu().numpy().squeeze().transpose((1, 2, 0))
                        noisy_save = input_[j, :3].detach().cpu().numpy().squeeze().transpose((1, 2, 0))
                        target_save = target[j].detach().cpu().numpy().squeeze().transpose((1, 2, 0))
                        rgb_restored = convert_color_space(restored_save, opt.color_space, 'rgb')
                        rgb_noisy = convert_color_space(noisy_save, opt.color_space, 'rgb')
                        rgb_target = convert_color_space(target_save, opt.color_space, 'rgb')
                        utils.save_img(rgb_restored*255.0, os.path.join(result_dir, filename+"-psnr{:.2f}.png".format(psnr)))
                        utils.save_img(rgb_noisy*255.0, os.path.join(result_dir, filename+"-input.png"))
                        utils.save_img(rgb_target*255.0, os.path.join(result_dir, filename+"-gt.png"))
                        if opt.joint_learning_alpha:
                            mask_target_save = (mask[j] * 255).detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                            mask_pred_save = (restored_mask[j] * 255).detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                            utils.save_img(mask_target_save, os.path.join(result_dir, filename+"-mask_target.png"))
                            utils.save_img(mask_pred_save, os.path.join(result_dir, filename+"-mask_pred.png"))
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        #### Evaluation ####
        psnr_val_rgb = 0
        if (index+1)%eval_now==0 and i>0:
            eval_shadow_rmse = 0
            eval_nonshadow_rmse = 0
            eval_rmse = 0
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                result_dir = os.path.join(log_dir, 'results', str(epoch))
                os.makedirs(result_dir, exist_ok=True)
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    if opt.w_hsv:
                        hsv = rgb_to_hsv(input_)
                        input_ = torch.cat((input_, hsv), dim=1)
                    if opt.joint_learning_alpha:
                        # mask_number_per = data[4].cuda()
                        # canny = kornia.filters.Canny()
                        # _, mask_edge = canny(mask)
                        # mask_edge = mask_edge.cuda()
                        mask_number_per = None
                        mask_edge = None
                    filenames = data_val[3]
                    with torch.cuda.amp.autocast():
                        if opt.joint_learning_alpha:
                            restored, restored_mask, loss_shadow, feature_input = model_restoration(input_, mask, mask_edge, mask_number_per)
                        else:
                            restored, feature_input = model_restoration(input_, mask)
                    if opt.color_space == 'hsv':
                        restored[:, 0] = input_[:, 0]
                        restored[:, 1] = input_[:, 1]
                        restored[:, 2] = torch.clamp(restored[:, 2],0,1)
                    else:
                        restored = torch.clamp(restored,0,1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False, color_space=opt.color_space).item())
                    if filenames[0] in ['0902.png', '0908.png', '0960.png', '0989.png']:
                        restored = restored.cpu().numpy().squeeze().transpose((1, 2, 0))
                        rgb_restored = convert_color_space(restored, opt.color_space, 'rgb')
                        utils.save_img(rgb_restored*255.0, os.path.join(result_dir, filenames[0]))
                        if opt.joint_learning_alpha:
                            mask_pred_save = (restored_mask[0] * 255).detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                            utils.save_img(mask_pred_save, os.path.join(result_dir, filenames[0]+"-mask_pred.png"))
    
                psnr_val_rgb = sum(psnr_val_rgb)/len(val_dataset)
                wandb.log({"val_psnr":psnr_val_rgb, 'epoch': epoch})
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
                print("\n[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    
    now = datetime.datetime.now()
    timestamp = int(now.timestamp())
    psnr = sum(psnr_list) / len(train_dataset)
    line_log = ""
    line_log += f"TimeStamp: {now.strftime('%Y-%m-%d %H:%M:%S')}\tEpoch: {epoch}\tTime: {time.time() - epoch_start_time:.3f}\tPSNR: {psnr:.3f}\tLearningRate {scheduler.get_lr()[0]:.6f}\nLoss: {epoch_loss['sum']:.4f}\t"
    if epoch_loss['crite']:
        line_log += f"(crite): {epoch_loss['crite']:.4f}\t"
    if epoch_loss['dino']:
        line_log += f"(dino): {epoch_loss['dino']:.4e}\t"
    if epoch_loss['seam']:
        line_log += f"(seam): {epoch_loss['seam']:.4e}\t"
    if epoch_loss['self_rep']:
        line_log += f"(self_rep): {epoch_loss['self_rep']:.3e}\t"
    if epoch_loss['ft']:
        line_log += f"(ft): {epoch_loss['ft']:.3e}\t"    
    if epoch_loss['shadow']:
        line_log += f"(shadow): {epoch_loss['shadow']:.3e}\t"    
    # line_log += f"LearningRate {scheduler.get_lr()[0]:.6f}"

    print("------------------------------------------------------------------")
    # print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
    print(line_log)
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        # f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')
        f.write(line_log.replace('\n', '\t'))
        f.write('\n')

    wandb_log = epoch_loss.copy()
    wandb_log["train_psnr"] = psnr
    # if psnr_val_rgb:
    #     wandb_log["val_psnr"] = psnr_val_rgb
    wandb_log["LearningRate"] = scheduler.get_lr()[0]
    wandb_log['epoch'] = epoch
    wandb.log(wandb_log)

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())



