import os
import torch
import argparse

# https://qiita.com/Hi-king/items/de960b6878d6280eaffc
class ParamProcessor(argparse.Action):
    """
    --param foo=a型の引数を辞書に入れるargparse.Action
    """
    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace,self.dest,[])
        if param_dict is None:
            param_dict = {}

        k, v = values.split("=")
        param_dict[k] = v
        setattr(namespace, self.dest, param_dict)


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
        parser.add_argument('--pretrain_weights',type=str, default='', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='2,3,0', help='GPUs')
        parser.add_argument('--arch', type=str, default ='ShadowFormer',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='shadow',  help='image restoration mode')
        parser.add_argument('--m_diff_alpha', type=float, default=0, help='diff image loss l1 weight')
        parser.add_argument('--m_shadow_alpha', type=float, default=0, help='shadow mask loss l1 weight')
        parser.add_argument('--dino_lambda', type=float, default=1e6, help='dino')
        parser.add_argument('--seam_lambda', type=float, default=0, help='seam')
        parser.add_argument('--seam_condition', action=ParamProcessor,
                            default={'loss_type':"ssim", 'edge_detector':"canny", 'color_space':"hsv"}, 
                            help="Arguments for seam loss can be specified. There are ['loss_type', 'edge_detector', 'color_space'] keys."
                            )
        parser.add_argument('--color_space', type=str, default ='rgb',  
                            choices=['rgb', 'bray', 'hsv', 'lab', 'luv', 'hls', 'yuv', 'xyz', 'ycrcb'], help='color space')
        parser.add_argument('--self_rep_lambda', type=float, default=0, help='weight of self-representation loss. When it is 0, this loss is not used.')
        parser.add_argument('--self_rep_once', action='store_true', default=False, help='backward default mse loss and self_rep loss same time')
        parser.add_argument('--self_feature_lambda', type=float, default=0, help='weight of feature loss')
        parser.add_argument('--mask_dir',type=str, default='mask_v_mtmt', help='mask directory')
        parser.add_argument('--cut_shadow_ratio',  type=float, default=0.5, help='percentage of cut shadow applied')
        parser.add_argument('--cut_shadow_ns_s_ratio', type=float, default=0, help='影なしに影あり : 影ありに影なし = (1 - ns_s_ratio) : ns_s_ratio')
        parser.add_argument('--nomixup', action='store_true', default=False, help='if you dont need mixup')
        parser.add_argument('--w_hsv', action='store_true', default=False, help='Add hsv to the input channel rgb')
        parser.add_argument('--sample_from_s', action='store_true', default=False, help='get pathes from shadow region whem cut shadow')
        parser.add_argument('--joint_learning_alpha', type=float, default=0, help='joint learning ratio. loss = loss_shadow * joint_learning_alpha + loss_other * (1 - joint_learning_alpha')
        parser.add_argument('--wo_wandb', action='store_true', default=False, help='if you dont need wandb')
        
        # MTMT
        parser.add_argument('--mtmt_pretrain_weights',type=str, default='./mtmt_model/weights/official_fine.pth', help='path of mtmt pretrained_weights')
        parser.add_argument('--mtmt_edge', type=float, default=10, help='edge learning weight')
        parser.add_argument('--mtmt_subitizing', type=float,  default=1, help='subitizing loss weight')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./log',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
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
