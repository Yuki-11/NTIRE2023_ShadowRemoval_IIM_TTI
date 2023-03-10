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

    def aug(self, rgb_gt, rgb_noisy, gray_mask):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        gray_mask2 = gray_mask[indices]
        # gray_contour2 = gray_mask[indices]
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
        gray_mask = lam * gray_mask + (1-lam) * gray_mask2
        # gray_mask = torch.where(gray_mask>0.01, torch.ones_like(gray_mask), torch.zeros_like(gray_mask))
        # gray_contour = lam * gray_contour + (1-lam) * gray_contour2
        return rgb_gt, rgb_noisy, gray_mask


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
    def __init__(self, p: float = 1.0, alpha: float = 0.7):
        """
        Parameters
        ----------
        p : float
            CutMixを実行する確率．
        alpha : float
            画像をくり抜くサイズのバイアス．
        """
        self.p = p
        self.alpha = alpha

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
        h, w = clean.size()[1:]
        ch, cw = int(h*cut_ratio), int(w*cut_ratio)

        fcy = np.random.randint(0, h-ch+1)
        fcx = np.random.randint(0, w-cw+1)
        tcy, tcx = fcy, fcx

        # if np.random.rand(1) > 0.5:
        clean[:, tcy:tcy+ch, tcx:tcx+cw] = noisy[:, fcy:fcy+ch, fcx:fcx+cw]
        mix_img = clean
        new_mask = torch.zeros_like(mask)
        new_mask[tcy:tcy+ch, tcx:tcx+cw] = mask[tcy:tcy+ch, tcx:tcx+cw]
        # else:
        #     noisy[:, tcy:tcy+ch, tcx:tcx+cw] = clean[:, fcy:fcy+ch, fcx:fcx+cw]
        #     mix_img = noisy
        #     mask[tcy:tcy+ch, tcx:tcx+cw] = 0.
        #     new_mask = mask
        
        return mix_img, new_mask
        