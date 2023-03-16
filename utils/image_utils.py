import torch
import numpy as np
import pickle
import cv2
from skimage.color import rgb2lab

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath, color_space='rgb'):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    img = convert_color_space(img, 'rgb', color_space)
    return img

def load_val_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def load_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # kernel = np.ones((8,8), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # contour = dilation - erosion
    img = img.astype(np.float32)
    # contour = contour.astype(np.float32)
    # contour = contour/255.
    img = img/255.
    return img

def load_diff(filepath):
    img = cv2.imread(filepath)
    img = img.astype(np.float32)
    img = img/255.
    return img

def load_val_mask(filepath):
    img = cv2.imread(filepath, 0)
    resized_img = img
    # resized_img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def save_img(img, filepath, color_space='rgb'):
    cvt_img = convert_color_space(img, color_space, 'rgb')
    if img.shape[2] > 1:
        cv2.imwrite(filepath, cv2.cvtColor(cvt_img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(filepath, img)

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def myPSNR_np(tar_img, prd_img):
    imdff = np.clip(prd_img,0,1) - np.clip(tar_img,0,1)
    rmse = np.sqrt((imdff**2).mean())
    ps = 20*np.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True, color_space='rgb'):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        # im1_cvt = convert_color_space(im1.cpu().numpy().transpose((1, 2, 0)), color_space, 'rgb')
        # im2_cvt = convert_color_space(im2.cpu().numpy().transpose((1, 2, 0)), color_space, 'rgb')
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):

        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy =
    return np.clip(image_numpy, 0, 255).astype(imtype)

def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

# def yCbCr2rgb(input_im):
#     im_flat = input_im.contiguous().view(-1, 3).float()
#     mat = torch.tensor([[1.164, 1.164, 1.164],
#                        [0, -0.392, 2.017],
#                        [1.596, -0.813, 0]])
#     bias = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0])
#     temp = (im_flat + bias).mm(mat)
#     out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])

def convert_color_space(img: np, from_space='rgb', to_space='hsv'):
    if from_space == to_space:
        return img
    enable_list = [
        'bray',
        'hsv',
        'lab',
        'luv',
        'hls',
        'yuv',
        'xyz',
        'ycrcb'
    ]
    assert ((from_space == 'rgb' and to_space in enable_list)
            or (from_space in enable_list and to_space == 'rgb')), \
            f'from_space: {from_space}, to_space: {to_space}, enable_list: {enable_list}'

    rgb2x_dict = {
        'bray': cv2.COLOR_RGB2GRAY,
        'hsv': cv2.COLOR_RGB2HSV,
        'lab': cv2.COLOR_RGB2Lab,
        'luv': cv2.COLOR_RGB2Luv,
        'hls': cv2.COLOR_RGB2HLS,
        'yuv': cv2.COLOR_RGB2YUV,
        'xyz': cv2.COLOR_RGB2XYZ,
        'ycrcb': cv2.COLOR_RGB2YCrCb,
    }
    x2rgb_dict = {
        'bray': cv2.COLOR_GRAY2RGB,
        'hsv': cv2.COLOR_HSV2RGB,
        'lab': cv2.COLOR_Lab2RGB,
        'luv': cv2.COLOR_Luv2RGB,
        'hls': cv2.COLOR_HLS2RGB,
        'yuv': cv2.COLOR_YUV2RGB,
        'xyz': cv2.COLOR_XYZ2RGB,
        'ycrcb': cv2.COLOR_YCrCb2RGB,
    }
    if from_space == 'rgb':
        return cv2.cvtColor(img, rgb2x_dict[to_space])
    else:
        return cv2.cvtColor(img, x2rgb_dict[from_space])

# https://github.com/rockeyben/DCCF/blob/master/iharm/model/ps_filters.py#L338-L390
def hsv_to_rgb(hsv):
    h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
    #对出界值的处理
    s = torch.clamp(s,0,1)
    v = torch.clamp(v,0,1)
    
    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))
    
    hi0 = hi==0
    hi1 = hi==1
    hi2 = hi==2
    hi3 = hi==3
    hi4 = hi==4
    hi5 = hi==5

    r = v * hi0 + q * hi1 + p * hi2 + p * hi3 + t * hi4 + v * hi5
    g = t * hi0 + v * hi1 + v * hi2 + q * hi3 + p * hi4 + p * hi5
    b = p * hi0 + p * hi1 + t * hi2 + v * hi3 + v * hi4 + q * hi5
    
    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    
    return rgb

# https://github.com/rockeyben/DCCF/blob/master/iharm/model/ps_filters.py#L338-L390
def rgb_to_hsv(img):
    eps = 1e-7
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
    saturation[ img.max(1)[0]==0 ] = 0

    value = img.max(1)[0]
    
    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value],dim=1)

    return hsv