
import os
import numpy as np
import math
import cv2
from PIL import Image
import time
import lpips
import torchvision.transforms as transforms
import torch

def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0 * 255.0 / mse)


def mse(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    return mse

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def lpips(result_path,GT_path):

    result = cv2.imread(result_path)
    GT = cv2.imread(GT_path)
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex', version=0.1)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg',
                              version=0.1)  # closer to "traditional" perceptual loss, when used for optimization
    test1_res = result
    test1_label = GT
    transf = transforms.ToTensor()
    test1_label = transf(test1_label)
    test1_res = transf(test1_res)
    test1_ress = test1_res.to(torch.float32).unsqueeze(0)
    test1_labell = test1_label.to(torch.float32).unsqueeze(0)
    lpips_loss = loss_fn_vgg(test1_ress, test1_labell)
    return lpips_loss.detach().numpy()

def psnr_ssim():
    #path1 = r'./data/SICE_light_test/out/'
    path1 = r'./out_muti1/'
    path2 = r'./out_muti2/'  # 指定原图文件夹

    #path1 =r'/home/ww/whd/contrast_test/mutilight_test/out/'
    #path2 = r'/home/ww/whd/contrast_test/mutilight_test/GT/'
    high_list = os.listdir(path1)

    list_psnr = []
    list_ssim = []
    list_mse = []
    list_lpips = []

    for i in high_list:

        img_a = cv2.imdecode(np.fromfile(path1 + i,dtype=np.uint8),-1)
        img_b = cv2.imdecode(np.fromfile(path2 + i,dtype=np.uint8),-1)

        psnr_num = psnr(img_a, img_b)
        ssim_num = ssim(img_a, img_b)
        '''mse_num = mse(img_a, img_b)
        lpips_nunm = lpips(path1 + i,path2 + i)'''
        #print(i , psnr_num)
        list_ssim.append(ssim_num)
        list_psnr.append(psnr_num)
        #list_mse.append(mse_num)
        #list_lpips.append(lpips_nunm)
        print(i ,psnr_num)
        print(i,ssim_num)




    return np.mean(list_psnr),np.mean(list_ssim)#,np.mean(list_mse),np.mean(list_lpips)
    #return np.mean(list_psnr)#,np.mean(list_ssim)


if __name__ == '__main__':
    #mean_psnr,mean_ssim,mean_mse,mean_lpips = psnr_ssim()
    mean_psnr = psnr_ssim()
    print("mean PSNR:", mean_psnr)  # ,list_psnr)
    print("mean SSIM:", mean_ssim)  # ,list_ssim)
    print("mean MSE:", mean_mse)  # ,list_mse)
    print("mean LPIPS:",mean_lpips)
