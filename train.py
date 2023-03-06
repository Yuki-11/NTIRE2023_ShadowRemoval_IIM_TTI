import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

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
import torch.optim as optim
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
from losses import CharbonnierLoss, DINOLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data
from utils.image_utils import convert_color_space, imsave


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
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
#     lr = utils.load_optim(optimizer, path_chk_rest)
#
#     for p in optimizer.param_groups: p['lr'] = lr
#     warmup = False
#     new_lr = lr
#     print('------------------------------------------------------------------------------')
#     print("==> Resuming Training with learning rate:",new_lr)
#     print('------------------------------------------------------------------------------')
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

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


######### Loss ###########
criterion = CharbonnierLoss(m_diff_alpha=opt.m_diff_alpha, m_shadow_alpha=opt.m_shadow_alpha, color_space=opt.color_space).cuda()
dino = DINOLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train, color_space=opt.color_space, mask_dir=opt.mask_dir)
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
    epoch_loss = {'crite': 0, 'dino': 0, 'self_rep': 0, 'sum': 0}
    train_id = 1
    epoch_ssim_loss = 0
    pbar = tqdm(train_loader, ncols=100)
    for i, data in enumerate(pbar, 0): 
        epoch_loss_formatted = dict()
        for key, value in epoch_loss.items():
            if key in ['dino']:
                epoch_loss_formatted[key] = f"{value:.1e}"
            else:
                epoch_loss_formatted[key] = f"{value:.1f}"
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
        if epoch > 5:
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)

        # self-representation learning 
        # if opt.self_rep_lambda:
        #     loss_self = 0.0
        #     with torch.cuda.amp.autocast():
        #         target_mask = torch.zeros_like(mask).cuda()
        #         restored = model_restoration(target, target_mask)
        #         restored = torch.clamp(restored,0,1)
        #         if opt.color_space == 'hsv':
        #             loss_self = loss_self + criterion(restored[:, 2], target[:, 2])
        #         else:
        #             loss_self = loss_self + criterion(restored, target, diff)
                
        #         if opt.dino_lambda:
        #             loss_dino = dino(restored, target)
        #             loss_self = loss_self + opt.dino_lambda * loss_dino
        #     epoch_loss['self_rep'] += loss_self.item()
        #     loss_self = loss_self * opt.self_rep_lambda
        #     epoch_loss['sum'] += loss_self.item()
        #     loss_scaler(
        #             loss_self, optimizer,parameters=model_restoration.parameters())

        with torch.cuda.amp.autocast():
            if opt.self_rep_lambda:
                loss_self = 0.0
                target_mask = torch.zeros_like(mask).cuda()
                restored = model_restoration(target, target_mask)
                restored = torch.clamp(restored,0,1)
                if opt.color_space == 'hsv':
                    loss_self = loss_self + criterion(restored[:, 2], target[:, 2])
                else:
                    loss_self = loss_self + criterion(restored, target, diff)
                if opt.dino_lambda:
                    loss_dino = dino(restored, target)
                    loss_self = loss_self + opt.dino_lambda * loss_dino
                epoch_loss['self_rep'] += loss_self.item()
                loss = loss + loss_self * opt.self_rep_lambda

            restored = model_restoration(input_, mask)
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
            epoch_loss['sum'] += loss.item()
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        #### Evaluation ####
        if (index+1)%eval_now==0 and i>0:# or True:
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
                    filenames = data_val[3]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_, mask)
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
    
                psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
                print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    
    line_log = ""
    line_log += f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.3f}\tLearningRate {scheduler.get_lr()[0]:.6f}\nLoss: {epoch_loss['sum']:.4f}\t"
    if epoch_loss['crite']:
        line_log += f"(crite): {epoch_loss['crite']:.4f}\t"
    if epoch_loss['dino']:
        line_log += f"(dino): {epoch_loss['dino']:.4e}\t"
    if epoch_loss['self_rep']:
        line_log += f"(self_rep): {epoch_loss['self_rep']:.3e}\t"
    # line_log += f"LearningRate {scheduler.get_lr()[0]:.6f}"

    print("------------------------------------------------------------------")
    # print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
    print(line_log)
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        # f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')
        f.write(line_log.replace('\n', '\t'))
        f.write('\n')

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



