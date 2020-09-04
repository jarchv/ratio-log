from PIL import Image
from skimage.metrics import structural_similarity

import numpy as np

def get_mse(x,y):
    return np.mean(np.square(x - y))

def get_psnr(inp_img,tar_img):
    inp_img_ar = np.asarray(inp_img, np.float32) / 255.0
    tar_img_ar = np.asarray(tar_img, np.float32) / 255.0 
    
    return 10. * np.log10(1. / get_mse(inp_img_ar, tar_img_ar))

def get_ssim(inp_img,tar_img):
    inp_img_ar = np.array(inp_img)
    tar_img_ar = np.array(tar_img)

    return structural_similarity(
        im1=inp_img_ar, 
        im2=tar_img_ar, 
        multichannel=True, 
        gaussian_weights=True, 
        use_sample_covariance=False)

def get_loe(inp_img,out_img):
    inp_img_ar = np.asarray(inp_img, dtype=np.int32)
    out_img_ar = np.asarray(out_img, dtype=np.int32)

    L  = np.max(inp_img_ar, axis = 2)
    Le = np.max(out_img_ar, axis = 2)
    
    m, n = L.shape
    r = 50 / min(m,n)
    dm, dn = int(m * r), int(n * r)

    L  = Image.fromarray(np.asarray(L, dtype=np.uint8), mode = 'L')
    Le = Image.fromarray(np.asarray(Le, dtype=np.uint8), mode = 'L')

    DL  = L.resize([dn,dm],Image.ANTIALIAS)
    DLe = Le.resize([dn,dm],Image.ANTIALIAS)

    DL  = np.asarray(DL , dtype = np.int32)
    DLe = np.asarray(DLe, dtype = np.int32)

    RD  = np.zeros((dm, dn), dtype=np.int32)

    for x in range(dm):
        for y in range(dn):
            UL  = DL[x,y] >= DL
            ULe = DLe[x,y] >= DLe
            RD[x,y] = np.sum(np.bitwise_xor(UL, ULe))
    
    return np.mean(RD)
    
def get_measures(inp_file, tar_file, out_file):
    inp_img = Image.open(inp_file)
    tar_img = Image.open(tar_file)
    out_img = Image.open(out_file)

    SSIM = get_ssim(tar_img, out_img)
    PSNR = get_psnr(tar_img, out_img)
    LOE  = get_loe(inp_img, out_img)
    LOET = get_loe(tar_img, out_img)

    inp_img.close()
    tar_img.close()
    out_img.close()

    return SSIM, PSNR, LOE, LOET

