
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import scipy.io as sio
from utils.loader import get_test_data
from utils.image_utils import convert_color_space, rgb_to_hsv
import utils
import cv2
from model import UNet

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='datasets/official/test_final/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/test',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/ShadowFormer_istd/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=4, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--color_space', type=str, default ='rgb',  
                    choices=['rgb', 'bray', 'hsv', 'lab', 'luv', 'hls', 'yuv', 'xyz', 'ycrcb'], help='color space')
parser.add_argument('--self_feature_lambda', type=float, default=0, help='weight of feature loss')
parser.add_argument('--mask_dir',type=str, default='mask_v_mtmt', help='mask directory')
parser.add_argument('--w_hsv', action='store_true', default=False, help='Add hsv to the input channel rgb')
parser.add_argument('--joint_learning_alpha', type=float, default=0, help='joint learning ratio. loss = loss_shadow * joint_learning_alpha + loss_other * (1 - joint_learning_alpha')
parser.add_argument('--mtmt_pretrain_weights',type=str, default='', help='path of mtmt pretrained_weights')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--train_ps', type=int, default=640, help='patch size of training sample')
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)
if args.joint_learning_alpha:
    os.makedirs(f"{args.result_dir}/pred", exist_ok=True)
    os.makedirs(f"{args.result_dir}/pred-mask", exist_ok=True)

test_dataset = get_test_data(args.input_dir, color_space=args.color_space, mask_dir=args.mask_dir, opt=args)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 8 * args.win_size

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    rmse_val_rgb = []
    psnr_val_s = []
    ssim_val_s = []
    psnr_val_ns = []
    ssim_val_ns = []
    rmse_val_s = []
    rmse_val_ns = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        # rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy = data_test[1].cuda()
        if args.joint_learning_alpha:
            mask = None
        else:
            mask = data_test[2].cuda()
            mask = F.pad(mask, (0, padw, 0, padh), 'reflect')
        filenames = data_test[4]
        if args.joint_learning_alpha:
            mask_number_per = None
            mask_edge = None

        # Pad the input if not_multiple_of win_size * 8
        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
        if args.w_hsv:
            hsv = rgb_to_hsv(rgb_noisy)
            rgb_noisy = torch.cat((rgb_noisy, hsv), dim=1)

        if args.tile is None:
            if args.joint_learning_alpha:
                rgb_restored, restored_mask, loss_shadow, _ = model_restoration(rgb_noisy, mask, mask_edge, mask_number_per)
            else:
                rgb_restored, _ = model_restoration(rgb_noisy, mask)
        else:
            # test the image tile by tile
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch, _ = model_restoration(in_patch, mask_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            restored = E.div_(W)

        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        # Unpad the output
        rgb_restored = rgb_restored[:height, :width, :]
        if args.joint_learning_alpha:
            mask_pred_save = (restored_mask[0] * 255).detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
            utils.save_img(mask_pred_save, os.path.join(args.result_dir, "pred-mask", filenames[0]))
            utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, "pred", filenames[0]), color_space=args.color_space)
        else:
            utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, filenames[0]), color_space=args.color_space)
