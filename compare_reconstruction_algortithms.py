import itertools
import os

import numpy as np
import torch
import tqdm.contrib.concurrent
from torchmetrics.image import lpip, psnr, ssim, vif

from algorithms.admm import ADMM
from algorithms.fista import FISTA
from algorithms.gradient_descent import gradient_descent
from algorithms.nesterov import nesterov_gradient_descent
from utils import load_psf_and_image


def load_image(image):
    return np.load(image)


algorithms = {
    'fista': FISTA,
    'nesterov': nesterov_gradient_descent,
    'gradient_descent': gradient_descent,
    'admm': ADMM,
}

n_iters = {
    'fista': 5000,
    'nesterov': 5000,
    'gradient_descent': 1000,
    'admm': 5000,
}

lensed_directory = os.path.join(os.getcwd(), 'data/lensed')
diffuser_directory = os.path.join(os.getcwd(), 'data/diffuser')


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

    toReturn = {"filename": filename,
                "LPIPS": float(lpips_value),
                "PSNR": float(psnr_value),
                "SSIM": float(ssim_value),
                "VIF": float(vif_value)}
    return toReturn


def reconstruct(filename, algorithm):
    psf, image = load_psf_and_image(psf_fp='data/psf.tiff',
                                    data_fp=os.path.join(diffuser_directory, filename),
                                    downsample=4, flip=True, normalize=True)
    gt = load_image(os.path.join(lensed_directory, filename))
    reconstructor = algorithms.get(algorithm)(psf=psf, gt=gt)
    return reconstructor.apply(image, disp_iter=n_iters.get(algorithm), verbose=False)[0]


def benchmark(args):
    filename, algorithm = args
    reconstructed = reconstruct(filename, algorithm)
    original = load_image(os.path.join(lensed_directory, filename))
    metrics = evaluate_reconstruction(filename, reconstructed, original)

    return {**metrics, 'algorithm': algorithm, 'reconstructed': reconstructed}


def main():
    images = ['im4527.npy', 'im16499.npy', 'im12612.npy', 'im12990.npy', 'im5035.npy']
    algorithms = ['fista', 'nesterov', 'gradient_descent', 'admm']

    results = tqdm.contrib.concurrent.thread_map(benchmark, itertools.product(images, algorithms), max_workers=12)
    # results = []
    # for args in itertools.product(images, algorithms):
    #     results.append(benchmark(args))
    np.save('algorithm_picture_benchmarks.npy', np.array(results))


if __name__ == '__main__':
    main()
