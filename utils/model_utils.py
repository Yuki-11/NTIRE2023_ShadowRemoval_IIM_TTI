import torch
import torch.nn as nn
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights, opt=None):
    checkpoint = torch.load(weights)
    if (opt is not None) and opt.joint_learning_alpha:
        model_state_dict = model.state_dict()
        for key in checkpoint["state_dict"]:
            if key in model_state_dict:
                model_state_dict[key] = checkpoint["state_dict"][key]
                continue
            tmp_key = f"module.{key}"
            if tmp_key in model_state_dict:
                model_state_dict[tmp_key] = checkpoint["state_dict"][tmp_key]
                continue
            tmp_key = key[7:]
            if tmp_key in model_state_dict:
                model_state_dict[tmp_key] = checkpoint["state_dict"][tmp_key]
                continue
        model.load_state_dict(model_state_dict)
    else:
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except Exception as e:  # add 'module.'
            print(e)
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v

            try:
                model.load_state_dict(new_state_dict)
            except Exception as e:  # remove 'module.'
                print(e)
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k if 'module.' not in k else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
    

def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import UNet, ShadowFormer, ShadowFormerJointMTMT
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'ShadowFormer':
        if opt.joint_learning_alpha:
            model_restoration = ShadowFormerJointMTMT(img_size=opt.train_ps,embed_dim=opt.embed_dim,
                                            win_size=opt.win_size,token_projection=opt.token_projection,
                                            token_mlp=opt.token_mlp, opt=opt)
        else:
            model_restoration = ShadowFormer(img_size=opt.train_ps,embed_dim=opt.embed_dim,
                                            win_size=opt.win_size,token_projection=opt.token_projection,
                                            token_mlp=opt.token_mlp, opt=opt)
    else:
        raise Exception("Arch error!")

    return model_restoration