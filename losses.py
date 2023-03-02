import torch
import torch.nn as nn
import torch.nn.functional as F
import vision_transformer as vits


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, m_diff_alpha=0, m_shadow_alpha=0, color_space='rgb'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.m_diff_alpha = m_diff_alpha
        self.m_shadow_alpha = m_shadow_alpha
        self.color_space = color_space

    def forward(self, x, y, mask=0, diff=0):
        xy_diff = x - y
        # if self.color_space == 'hsv':
        #     xy_diff[:, :, 0] = torch.min(xy_diff[:, :, 0], torch.abs(x[:, :, 0] + 1 - y[:, :, 0]))
        #     xy_diff[:, :, 0] = torch.min(xy_diff[:, :, 0], torch.abs(x[:, :, 0] - 1 - y[:, :, 0]))
        A = torch.ones(*xy_diff.shape).cuda()
        # loss = torch.sum(torch.sqrt(xy_diff * xy_diff + self.eps))
        loss = torch.mean(torch.sqrt((A + self.m_diff_alpha * diff + self.m_shadow_alpha * mask) * (xy_diff * xy_diff) + (self.eps*self.eps)))
        return loss



class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5):
        """Computes the structural similarity (SSIM) index map between two images.

        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x, y, as_loss: bool = True):

        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x, y):

        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float):

        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d


class DINOLoss(nn.Module):
    def __init__(self, loss_type="mse"):
        super().__init__()
        # vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        patch_size = 16
        self.deno_model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0)
        for p in self.deno_model.parameters():
            p.requires_grad = False
        self.deno_model.eval()
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth")
        self.deno_model.load_state_dict(state_dict, strict=True)
        if loss_type == "mse":
            self.loss_func = self._mse_loss
        elif loss_type == "cs":
            self.loss_func = self._cs_loss

    def _cosine_sim(self, key):
        # 分子計算
        kk = torch.matmul(key, key.permute(0, 1, 3, 2))
        # 分母計算
        key_abs = torch.norm(key, p = 2, dim = 3).unsqueeze(dim = -1)
        kk_abs = torch.matmul(key_abs, key_abs.permute(0, 1, 3, 2))
        # 類似度計算
        return 1. - kk / kk_abs 

    def _mse_loss(self, key1, key2):
        return F.mse_loss(key1, key2) 
    
    def _cs_loss(self, key1, key2):
        s1 = self._cosine_sim(key1)
        s2 = self._cosine_sim(key2)  # [b, heads, n_patches, n_patches]
        return torch.norm(s1 - s2, p="fro")

    def forward(self, img1, img2):
        _, key1 = self.deno_model.get_last_selfattention(img1, return_key=True)
        _, key2 = self.deno_model.get_last_selfattention(img2, return_key=True)
        # [b, num_heads, num_patches+1, 64]. [1,6,10801,64]
        #+1はクラス識別などのトークン？
        b,c,h,w = img1.shape
        pixes = b*h*w
        loss = self.loss_func(key1, key2) / pixes
        return loss
    


