from losses import DINOLoss
import torch
# from torchinfo import summary
from torchvision import transforms as pth_transforms
from PIL import Image


batch_size=1
patch_size = 16

# summary(model=vits16, input_size=(batch_size, 3, 1980, 1440))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_model = DINOLoss()
# loss_model.to(device)

def read_img(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    return img

for i in range(900, 1000):
    # img1 = read_img(f"/home/kondo/shadow_removal/NTIRE2023_ShadowRemoval_IIM_TTI/datasets/official_warped/train/input/{str(i).zfill(4)}.png")
    # img2 = read_img(f"/home/kondo/shadow_removal/NTIRE2023_ShadowRemoval_IIM_TTI/datasets/official_warped/train/gt/{str(i).zfill(4)}.png")
    # img1 = read_img(f"/home/kondo/shadow_removal/NTIRE2023_ShadowRemoval_IIM_TTI/results/val_warped/{str(i).zfill(4)}.png")
    img2 = read_img(f"/home/kondo/shadow_removal/NTIRE2023_ShadowRemoval_IIM_TTI/datasets/official_warped/val/input/{str(i).zfill(4)}.png")
    img1 = read_img(f"/home/kondo/shadow_removal/NTIRE2023_ShadowRemoval_IIM_TTI/datasets/official_warped/val/gt/{str(i).zfill(4)}.png")
    # img2 = read_img(f"/home/kondo/shadow_removal/NTIRE2023_ShadowRemoval_IIM_TTI/results/val_warped/{str(i).zfill(4)}.png")

    # img1.to(device)
    # img2.to(device)
    # print(img1.device)
    # attentions = model.get_last_selfattention(img.to(device))

    loss = loss_model(img1, img2)

    print(i, loss)