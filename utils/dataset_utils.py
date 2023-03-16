import torch
import os

import numpy as np
import torchvision.transforms as T
from typing import Tuple, Optional

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy, gray_mask, gray_mask_edge=None):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        gray_mask2 = gray_mask[indices]
        if gray_mask_edge:
            gray_mask_edge2 = gray_mask_edge[indices]
        # gray_contour2 = gray_mask[indices]
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
        gray_mask = lam * gray_mask + (1-lam) * gray_mask2
        if gray_mask_edge:
            gray_mask_edge = lam * gray_mask_edge + (1-lam) * gray_mask_edge2
        # gray_mask = torch.where(gray_mask>0.01, torch.ones_like(gray_mask), torch.zeros_like(gray_mask))
        # gray_contour = lam * gray_contour + (1-lam) * gray_contour2
        return rgb_gt, rgb_noisy, gray_mask, gray_mask_edge


class CutShadow(object):
    """
    CutShadow
    一部分を影有り画像に変える

    Attributes
    ----------
    p : float
        CutMixを実行する確率．
    alpha : float
        画像をくり抜くサイズのバイアス．
    """
    def __init__(self, p: float = 1.0, alpha: float = 0.7, ns_s_ratio: float = 0.0, sample_from_s: bool = False):
        """
        Parameters
        ----------
        p : float
            CutMixを実行する確率．
        alpha : float
            画像をくり抜くサイズのバイアス．
        ns_s_ratin : float
            影なしに影あり : 影ありに影なし = (1 - ns_s_ratio) : ns_s_ratio
        sample_from_s : bool
            画像をくり抜く際に，影領域が中央に含まれるようにサンプリングする
        """
        self.p = p
        self.alpha = alpha
        self.ns_s_ratio = ns_s_ratio
        self.sample_from_s = sample_from_s

    def __call__(self, clean: torch.Tensor, noisy: torch.Tensor, mask: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        clean : torch.Tensor
            shadow-free image
        noisy : torch.Tensor
            shadow image
        mask : torch.Tensor
            mask of shadow image
    
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            mix_img, mask
        """
        if np.random.rand(1) >= self.p:
            return noisy, mask

        cut_ratio = np.random.randn() * 0.01 + self.alpha
        c, h, w = clean.size()
        ch, cw = int(h*cut_ratio), int(w*cut_ratio)

        if self.sample_from_s:
            random_mask = mask.detach().clone() * torch.rand(*clean.size()[1:])
            choice_flatten = random_mask.argmax()
            fcy, fcx = choice_flatten // w - ch//2, choice_flatten % w - cw//2
            my, mx = choice_flatten // w, choice_flatten % w
            # print("中心点xy:", mx, my)
            fcy, fcx = max(fcy, 0), max(fcx, 0)
            fcy, fcx = min(fcy, h-ch+1), min(fcx, w-cw+1)
        else:
            fcy = np.random.randint(0, h-ch+1)
            fcx = np.random.randint(0, w-cw+1)

        if np.random.rand(1) >= self.ns_s_ratio:
            mix_img = clean.detach().clone()
            mix_img[:, fcy:fcy+ch, fcx:fcx+cw] = noisy[:, fcy:fcy+ch, fcx:fcx+cw]
            new_mask = torch.zeros_like(mask)
            new_mask[fcy:fcy+ch, fcx:fcx+cw] = mask[fcy:fcy+ch, fcx:fcx+cw]
        else:
            mix_img = noisy.detach().clone()
            mix_img[:, fcy:fcy+ch, fcx:fcx+cw] = clean[:, fcy:fcy+ch, fcx:fcx+cw]
            mask[fcy:fcy+ch, fcx:fcx+cw] = 0.
            new_mask = mask
        
        # radius = 8
        # color = (1, 0, 0)
        # mix_img[:, my:my+3, mx:mx+3] = torch.tensor([1., 0., 1.])
        # mix_img[:, my-radius:my+radius, mx-radius:mx+radius] = torch.tensor(color).view(3, 1, 1)
        return mix_img, new_mask
        