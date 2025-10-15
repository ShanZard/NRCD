from torchvision.transforms import *
import numpy as np
import skimage.io as skio
import albumentations as A
import os 
import torch

tp = ToPILImage()
rp =RandomPerspective(p=1)
rr =RandomRotation(degrees=(-180,180))
gs = Grayscale()
cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)
gb = GaussianBlur(kernel_size=(7,7),sigma=(0.1,2)) 

od = A.OpticalDistortion(distort_limit=0.3,p=1)
sr = A.Solarize(threshold=(128, 255), p=1)
rf = A.RandomFog( p=1)
cd = A.CoarseDropout(p=1, max_holes=5, max_height=24, max_width=24, fill_value=0)

### More parameter settings can be found in the official library documentations of torchvision and albumentations

if __name__ == "__main__":
        ##### some augmentation you should use albumentations to do data augmentation #####
        ##### you juse change T2 images, and T1 images and labels remain the same #####
        T2_images_path = '/data/t2_images'
        ##### for example, we use gaussian CoarseDropout to change T2 images #####
        save_T2_images_path = '/data/t2_images_cd'
        os.makedirs(save_T2_images_path,exist_ok=True)
        for tmp in os.listdir(T2_images_path):
                imageB = skio.imread(os.path.join(T2_images_path,tmp))
                saveimage = cd(image = imageB)
                savefilepath = os.path.join(save_T2_images_path,tmp)
                saveimage = saveimage['image']
                skio.imsave(savefilepath,saveimage)
        
        #### some augmentation you should use torchvision to do data augmentation #####
        T2_images_path = '/data/t2_images'
        ##### for example, we use gaussian CoarseDropout to change T2 images #####
        save_T2_images_path = '/data/t2_images_gb'
        os.makedirs(save_T2_images_path,exist_ok=True)
        for tmp in os.listdir(T2_images_path):
                imageB = skio.imread(os.path.join(T2_images_path,tmp))
                imageB_tensor = torch.tensor(imageB).permute(2,0,1)
                saveimage_tensor = gb(imageB_tensor)
                saveimage = np.array(saveimage_tensor).transpose(1,2,0)
                savefilepath = os.path.join(save_T2_images_path,tmp)
                skio.imsave(savefilepath,saveimage)
