import matplotlib.pyplot as plt
import torch
from torchmetrics.image import lpip, psnr, ssim, vif

from algorithms.admm import ADMM
from utils import load_psf_and_image, to_absolute_path, plot_image, load_image

import warnings
warnings.filterwarnings('ignore')

import csv

import pandas as pd


def run_admm(filename):
    # downsample the psf by 4 because images are of size 270x480 while the PSF is of size 1920x1080

    ORIGINAL_IMAGE = "data/lensed/" + filename 
    DIFFUSER_IMAGE = "data/diffuser/"  + filename
    PSF = "data/psf.tiff"

    psf, image = load_psf_and_image(psf_fp=to_absolute_path(PSF), data_fp=to_absolute_path(DIFFUSER_IMAGE), downsample=4, flip=True, normalize=True)

    reconstructor = ADMM(psf)

    res = reconstructor.apply(image, disp_iter=50)
    reconstructed = res[0]
    #plt.show()

    original = load_image(to_absolute_path(ORIGINAL_IMAGE), shape=reconstructed.shape)
    #plot_image(original)
    #plt.title("Ground truth image")
    #plt.show()

    return evaluate_reconstruction(filename, reconstructed, original)


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
    print(f"LPIPS : {lpips_value}")
    print(f"PSNR : {psnr_value}")
    print(f"SSIM : {ssim_value}")
    print(f"VIF : {vif_value}")

    toReturn = {"Filename": filename, 
                "LPIPS": float(lpips_value), 
                "PSNR": float(psnr_value), 
                "SSIM": float(ssim_value), 
                "VIF": float(vif_value)}
    return toReturn


if __name__ == "__main__":

    #get files
    mypath = "/Users/ishapuri/Desktop/final_opt_project/lenseless_imaging/data/diffuser"
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    #get evaluation results
    results = []
    for filename in onlyfiles:
        if ".DS_Store" not in filename:
            print("FILENAME: ", filename)
            result = run_admm(filename)
            results.append(result)#results[filename] = result
    print(results)


    #save to CSV

    csv_filename = "test.csv"
    with open(csv_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filename", "LPIPS", "PSNR", "SSIM", "VIF"])
        writer.writeheader()
        writer.writerows(results)


    #calculate averages
    print("\n\n--AVERAGES--")
    df = pd.read_csv(csv_filename)[["LPIPS", "PSNR", "SSIM", "VIF"]]
    averages = df.mean()
    metrics = ["LPIPS", "PSNR", "SSIM", "VIF"]
    for i in range(0, 4): 
        metric = metrics[i]
        print(str(metric) + " Average: ", averages[i])


