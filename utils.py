import os

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from scipy.fftpack import next_fast_len


def load_psf_and_image(psf_fp, data_fp, return_float=True, downsample=None, bg_pix=(5, 25), flip=False, shape=None, normalize=False):
    """
    Load image for image reconstruction.

    Parameters
    ----------
    psf_fp : str
        Full path to PSF file.
    data_fp : str
        Full path to measurement file.
    return_float : bool, optional
        Whether to return PSF as float array, or unsigned int.
    downsample : int or float
        Downsampling factor for the PSF.
    bg_pix : tuple, optional
        Section of pixels to take from top left corner to remove background
        level. Set to `None` to omit this step, although it is highly
        recommended.
    flip : bool
        Whether to flip image (vertical and horizontal).
    normalize : bool default True
        Whether to normalize image to maximum value of 1.

    Returns
    -------
    psf : :py:class:`~numpy.ndarray`
        2-D array of PSF.
    image : :py:class:`~numpy.ndarray`
        2-D array of raw measurement image.
    """

    assert os.path.isfile(psf_fp)
    assert os.path.isfile(data_fp)
    if shape is None:
        assert downsample is not None

    # load and process PSF image
    psf, bg = load_psf(psf_fp, downsample=downsample, return_float=return_float, bg_pix=bg_pix, return_bg=True, flip=flip, shape=shape)

    # load and process raw measurement
    image = load_image(data_fp, flip=flip, bg=bg, return_float=return_float, shape=shape, normalize=normalize)

    psf = np.array(psf)
    image = np.array(image)
    return psf, image


def to_absolute_path(path: str) -> str:
    """
    Converts the specified path to be an absolute path.
    If the input path is relative, it's interpreted as relative to the current working directory.
    If it's absolute, it's returned as is.
    """
    # Check if the path is already absolute
    if os.path.isabs(path):
        return path
    else:
        # Combine the current working directory with the relative path
        return os.path.abspath(path)


def get_max_val(img):
    nbits = int(np.ceil(np.log2(img.max())))
    return 2 ** nbits - 1


def resize(img, factor=None, shape=None):
    """
    Resize by given factor.

    Parameters
    ----------
    img : :py:class:`~numpy.ndarray`
        Downsampled image.
    factor : int or float
        Resizing factor.
    shape : tuple
        Shape to copy ([depth,] height, width, color). If provided, (height, width) is used.

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        Resized image.
    """
    min_val = img.min()
    max_val = img.max()
    img_shape = np.array(img.shape)[-3:-1]

    assert not ((factor is None) and (shape is None)), "Must specify either factor or shape"
    new_shape = tuple(img_shape * factor) if shape is None else shape[-3:-1]
    new_shape = [int(i) for i in new_shape]

    if np.array_equal(img_shape, new_shape):
        return img

    # torch resize expects an input of form [color, depth, width, height]
    tmp = np.moveaxis(img, -1, 0)
    tmp = torch.from_numpy(tmp.copy())
    resized = torchvision.transforms.Resize(size=new_shape, antialias=True)(tmp).numpy()
    resized = np.moveaxis(resized, 0, -1)

    return np.clip(resized, min_val, max_val)


def plot_image(img, ax=None) -> plt.Axes:
    """
    Plot image data, correcting for gamma
    """

    # if we have only 1 depth, remove the axis
    if img.shape[0] == 1:
        img = img[0]

    disp_img = None
    cmap = None

    # full 3D RGB format : [depth, width, height, color]
    # data of length 3 means we have to infer whichever depth or color is missing, based on shape.
    if len(img.shape) == 3:
        disp_img = [img]
        cmap = None
    # data of length 2 means we have only width and height
    elif len(img.shape) == 2:  # 2D grayscale
        disp_img = [img]
        cmap = "gray"

    max_val = [d.max() for d in disp_img]

    assert len(disp_img) == 1 or len(disp_img) == 3

    # need float image for gamma correction and plotting
    img_norm = disp_img.copy()
    for i in range(len(img_norm)):
        img_norm[i] = disp_img[i] / max_val[i]

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(img_norm[0], cmap=cmap)

    return ax


def finite_diff_gram(shape):
    """Gram matrix of finite difference operator."""
    gram = np.zeros(shape, dtype=np.float32)
    gram[0, 0, 0] = 4
    gram[0, 0, 1] = gram[0, 0, -1] = gram[0, 1, 0] = gram[0, -1, 0] = -1

    return fft.rfft2(gram, axes=(-3, -2))


def finite_diff(x):
    """Gradient of image estimate, approximated by finite difference. Space where image is assumed sparse."""
    return np.stack((np.roll(x, 1, axis=-3) - x, np.roll(x, 1, axis=-2) - x), axis=len(x.shape), )


def finite_diff_adj(x):
    """Adjoint of finite difference operator."""
    diff1 = np.roll(x[..., 0], -1, axis=-3) - x[..., 0]
    diff2 = np.roll(x[..., 1], -1, axis=-2) - x[..., 1]
    return diff1 + diff2


def soft_thresh(x, thresh):
    return np.sign(x) * np.maximum(0, np.abs(x) - thresh)


def load_image(fp, flip=True, downsample=None, bg=None, return_float=False, shape=None, dtype=None, normalize=True, ):
    """
    Load image as numpy array.

    Parameters
    ----------
    fp : str
        Full path to file.
    flip : bool
        Whether to flip data (vertical and horizontal).
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    bg : array_like
        Background level to subtract.
    return_float : bool
        Whether to return image as float array, or unsigned int.
    shape : tuple, optional
        Shape (H, W, C) to resize to.
    dtype : str, optional
        Data type of returned data. Default is to use that of input.
    normalize : bool, default True
        If ``return_float``, whether to normalize data to maximum value of 1.

    Returns
    -------
    img : :py:class:`~numpy.ndarray`
        RGB image of dimension (height, width, 3).
    """
    assert os.path.isfile(fp)

    if "npy" in fp or "npz" in fp:
        img = np.load(fp)
    else:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_dtype = img.dtype

    if flip:
        img = np.flipud(img)
        img = np.fliplr(img)

    if bg is not None:
        img = img - bg
        img = np.clip(img, a_min=0, a_max=img.max())

    if downsample is not None or shape is not None:
        if downsample is not None:
            factor = 1 / downsample
        else:
            factor = None
        img = resize(img, factor=factor, shape=shape)

    if return_float:
        if dtype is None:
            dtype = np.float32
        assert dtype == np.float32 or dtype == np.float64
        img = img.astype(dtype)
        if normalize:
            img /= img.max()

    else:
        if dtype is None:
            dtype = original_dtype
        img = img.astype(dtype)

    return img


def load_psf(fp, downsample=1, return_float=True, bg_pix=(5, 25), return_bg=False, flip=False, dtype=np.float32, shape=None):
    """
    Load and process PSF for analysis or for reconstruction.

    Basic steps are:
    * Load image.
    * (Optionally) subtract background. Recommended.
    * (Optionally) resize to more manageable size
    * (Optionally) normalize within [0, 1] if using for reconstruction; otherwise cast back to uint for analysis.

    Parameters
    ----------
    fp : str
        Full path to file.
    downsample : int, optional
        Downsampling factor. Recommended for image reconstruction.
    return_float : bool, optional
        Whether to return PSF as float array, or unsigned int.
    bg_pix : tuple, optional
        Section of pixels to take from top left corner to remove background level. Set to `None` to omit this
        step, althrough it is highly recommended.
    return_bg : bool, optional
        Whether to return background level, for removing from data for reconstruction.
    flip : bool, optional
        Whether to flip up-down and left-right.
    Returns
    -------
    psf : :py:class:`~numpy.ndarray`
        4-D array of PSF.
    """

    # load image data and extract necessary channels
    psf = load_image(fp, flip=flip)

    max_val = get_max_val(psf)
    psf = np.array(psf, dtype=dtype)

    psf = psf[np.newaxis, :, :, :]

    # check that all depths of the psf have the same shape.
    for i in range(len(psf)):
        assert psf[0].shape == psf[i].shape

    # compute the background level to subtract
    bg = []
    for i in range(psf.shape[3]):
        bg_i = np.mean(psf[:, bg_pix[0]: bg_pix[1], bg_pix[0]: bg_pix[1], i])
        psf[:, :, :, i] -= bg_i
        bg.append(bg_i)

    psf = np.clip(psf, a_min=0, a_max=psf.max())
    bg = np.array(bg)

    # resize
    if downsample != 1:
        psf = resize(psf, shape=shape, factor=1 / downsample)

    # normalize
    if return_float:
        psf /= np.linalg.norm(psf.ravel())
        bg /= max_val
    if return_bg:
        return psf, bg


class RealFFTConvolve2D:
    def __init__(self, psf, norm="ortho"):
        """
        Linear operator that performs convolution in Fourier domain, and assumes
        real-valued signals.

        Parameters
        ----------
        psf :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that algorithms forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        norm : str, optional
            Normalization to use for FFT. Defaults to 'ortho'.
        """

        assert len(psf.shape) == 4, "Expected 4D PSF of shape (depth, width, height, channels)"
        assert psf.shape[3] == 3 or psf.shape[3] == 1

        self._psf = psf.astype(np.float32)
        self._psf_shape = np.array(self._psf.shape)

        # cropping / padding indexes
        self._padded_shape = 2 * self._psf_shape[-3:-1] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = list(np.r_[self._psf_shape[-4], self._padded_shape, self._psf_shape[-1]])
        self._start_idx = (self._padded_shape[-3:-1] - self._psf_shape[-3:-1]) // 2
        self._end_idx = self._start_idx + self._psf_shape[-3:-1]

        # precompute filter in frequency domain
        # self._H = fft.rfft2(self.pad(self._psf), axes=(-3, -2), norm=norm)
        self._H = torch.fft.rfftn(torch.from_numpy(self.pad(self._psf)), dim=(-3, -2), norm=norm)
        self._Hadj = torch.conj(self._H)
        self._padded_data = np.zeros(self._padded_shape)

    def crop(self, x):
        return x[..., self._start_idx[0]: self._end_idx[0], self._start_idx[1]: self._end_idx[1], :]

    def pad(self, v):
        if len(v.shape) == 5:
            batch_size = v.shape[0]
        elif len(v.shape) == 4:
            batch_size = 1
        else:
            raise ValueError("Expected 4D or 5D tensor")
        shape = [batch_size] + self._padded_shape
        vpad = np.zeros(shape).astype(v.dtype)
        vpad[..., self._start_idx[0]: self._end_idx[0], self._start_idx[1]: self._end_idx[1], :] = v
        return vpad

    def convolve(self, x):
        # # check type of x
        # if isinstance(x, np.ndarray):
        #     x = torch.from_numpy(x)

        # Real-to-complex Fourier transform
        data_ft = torch.fft.rfftn(x, dim=(-3, -2))

        # Multiplication in Fourier domain
        conv_result_ft = data_ft * self._H

        # Inverse complex-to-real Fourier transform
        conv_result = torch.fft.irfftn(conv_result_ft, s=x.shape[-3:-1], dim=(-3, -2))

        # Shifting the zero frequency component
        conv_result = torch.fft.fftshift(conv_result, dim=(-3, -2))

        return conv_result

    def deconvolve(self, y):
        """ Deconvolve with adjoint of pre-computed FFT of provided PSF. """
        # Convert y to PyTorch tensor if it is not already
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)

        # Real-to-complex Fourier transform
        padded_data_ft = torch.fft.rfftn(y, dim=(-3, -2))

        # Multiplication in Fourier domain with the adjoint
        deconv_result_ft = padded_data_ft * self._Hadj

        # Inverse complex-to-real Fourier transform
        deconv_result = torch.fft.irfftn(deconv_result_ft, s=y.shape[-3:-1], dim=(-3, -2))

        # Shifting the zero frequency component
        deconv_output = torch.fft.fftshift(deconv_result, dim=(-3, -2))

        return deconv_output
