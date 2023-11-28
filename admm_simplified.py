import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image import lpip, psnr

from models.admm import ADMM
from models.reconstructor import ReconstructionAlgorithm
from utils import load_psf_and_image, to_absolute_path, plot_image, load_image

ORIGINAL_IMAGE = "data/lensed/im172.npy"
DIFFUSER_IMAGE = "data/diffuser/im172.npy"
PSF = "data/psf.tiff"

def run_admm():
    # downsample the psf by 4 because images are of size 270x480 while the PSF is of size 1920x1080
    psf, image = load_psf_and_image(psf_fp=to_absolute_path(PSF), data_fp=to_absolute_path(DIFFUSER_IMAGE), downsample=4, flip=True, normalize=True)

    reconstructor = ADMM(psf)

    res = reconstructor.apply(image, disp_iter=50)
    img = res[0]

    plt.show()

    original = load_image(to_absolute_path(ORIGINAL_IMAGE), shape=img.shape)
    plot_image(original)
    plt.title("Ground truth image")
    plt.show()

    # compute metrics
    lpips_func = lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    psnr_funct = psnr.PeakSignalNoiseRatio()

    img_torch = torch.from_numpy(img).float()  # TODO
    original_torch = torch.from_numpy(original).unsqueeze(0)

    # channel as first dimension
    img_torch = img_torch.movedim(-1, -3)
    original_torch = original_torch.movedim(-1, -3)

    # normalize, TODO img max value is 14 which seems strange
    img_torch = img_torch / torch.amax(img_torch)

    # compute metrics
    lpips = lpips_func(img_torch, original_torch)
    psnr_value = psnr_funct(img_torch, original_torch)
    print(f"LPIPS : {lpips}")
    print(f"PSNR : {psnr_value}")


if __name__ == "__main__":
    run_admm()
