import multiprocessing
import os
import time
import warnings
from os import listdir
from os.path import isfile, join

import torch
import tqdm.contrib
import tqdm.contrib.concurrent
from torchmetrics.image import lpip, psnr, ssim, vif

from algorithms.admm import ADMM
from utils import load_psf_and_image, to_absolute_path, load_image

warnings.filterwarnings('ignore')

import csv

import pandas as pd


def run_admm(filename):
    # downsample the psf by 4 because images are of size 270x480 while the PSF is of size 1920x1080

    ORIGINAL_IMAGE = "data/lensed/" + filename
    DIFFUSER_IMAGE = "data/diffuser/" + filename
    PSF = "data/psf.tiff"

    psf, image = load_psf_and_image(psf_fp=to_absolute_path(PSF), data_fp=to_absolute_path(DIFFUSER_IMAGE), downsample=4, flip=True, normalize=True)

    reconstructor = ADMM(psf)

    res = reconstructor.apply(image, disp_iter=1000)
    reconstructed = res[0]

    original = load_image(to_absolute_path(ORIGINAL_IMAGE), shape=reconstructed.shape)

    return evaluate_reconstruction(filename, reconstructed, original)

    # metrics = evaluate_reconstruction(filename, reconstructed, original)
    # losses = reconstructor.get_losses()
    #
    # return {**metrics, 'loss': losses}



def evaluate_reconstruction(filename, reconstructed, original):
    # compute metrics
    lpips_func = lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    psnr_funct = psnr.PeakSignalNoiseRatio()
    ssim_funct = ssim.StructuralSimilarityIndexMeasure()
    vif_funct = vif.VisualInformationFidelity()

    img_torch = torch.from_numpy(reconstructed).float()
    original_torch = torch.from_numpy(original).unsqueeze(0)

    # channel as first dimension
    img_torch = img_torch.movedim(-1, -3)
    original_torch = original_torch.movedim(-1, -3)

    # normalize
    img_torch = img_torch / torch.amax(img_torch)

    # compute metrics
    lpips_value = lpips_func(img_torch, original_torch)
    psnr_value = psnr_funct(img_torch, original_torch)
    ssim_value = ssim_funct(img_torch, original_torch)
    vif_value = vif_funct(img_torch, original_torch)

    toReturn = {"Filename": filename,
                "LPIPS": float(lpips_value),
                "PSNR": float(psnr_value),
                "SSIM": float(ssim_value),
                "VIF": float(vif_value)}
    return toReturn


if __name__ == "__main__":
    startTime = time.time()

    # get current directory
    mypath = os.path.join(os.getcwd(), "data/diffuser")
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    results = tqdm.contrib.concurrent.process_map(run_admm, onlyfiles,
                                                  max_workers=multiprocessing.cpu_count())

    # save to CSV
    csv_filename = "test.csv"
    with open(csv_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filename", "LPIPS", "PSNR", "SSIM", "VIF"])
        writer.writeheader()
        writer.writerows(results)

    # calculate averages
    print("\n\n--AVERAGES--")
    df = pd.read_csv(csv_filename)[["LPIPS", "PSNR", "SSIM", "VIF"]]
    averages = df.mean()
    metrics = ["LPIPS", "PSNR", "SSIM", "VIF"]
    for i in range(0, 4):
        metric = metrics[i]
        print(str(metric) + " Average: ", averages[i])

    endTime = time.time()

    print("\nTime Taken: ", endTime - startTime, "s for ", len(onlyfiles), " images")
