import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=6, help='batch size')
        parser.add_argument('--nepoch', type=int, default=500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=16, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='ISTD')
        parser.add_argument('--pretrain_weights',type=str, default='./log/model_best.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='1,2,3,4,7,8,9', help='GPUs')
        parser.add_argument('--arch', type=str, default ='ShadowFormer',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='shadow',  help='image restoration mode')
        parser.add_argument('--m_diff_alpha', type=float, default=0, help='diff image loss l1 weight')
        parser.add_argument('--m_shadow_alpha', type=float, default=0, help='shadow mask loss l1 weight')
        parser.add_argument('--dino_lambda', type=float, default=0, help='dino')
        parser.add_argument('--color_space', type=str, default ='rgb',  
                            choices=['rgb', 'bray', 'hsv', 'lab', 'luv', 'hls', 'yuv', 'xyz', 'ycrcb'], help='color space')
        parser.add_argument('--self_rep_lambda', type=float, default=0, help='size of self-representation loss. When it is 0, this loss is not used.')
        
        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./log',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_m_shadow1',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=4, help='window size of self-attention') # default : 10
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
        
        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
        
        # args for training
        parser.add_argument('--train_ps', type=int, default=640, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default ='datasets/official_warped/train',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='datasets/official_warped/val',  help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        return parser
