import numpy as np
from scipy import fft

from algorithms.reconstruction_algorithm import ReconstructionAlgorithm
from utils import finite_diff_gram, finite_diff, finite_diff_adj, soft_thresh, RealFFTConvolve2D


# noinspection PyPep8Naming
class gradient_descent(ReconstructionAlgorithm):
    """
    Object for applying conventional gradient descent
    """

    def __init__(self, psf, norm="backward"):
        """
        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        norm : str
            Normalization to use for the convolution. Options are "forward",
            "backward", and "ortho". Default is "backward".
        """
        self._psf = psf.astype(np.float32)
        self._convolver = RealFFTConvolve2D(psf, norm=norm)
        self._padded_shape = self._convolver._padded_shape
        super(GD, self).__init__(psf, norm=norm)

    def reset(self):
        self._image_est = np.zeros(self._padded_shape, dtype=np.float32)
        self._step_size = np.real(1.8/np.max(self._convolver._H * self._convolver._Hadj))
        self._alpha = np.real(1.8 / np.max(self._convolver._Hadj * self._convolver._H, axis=0))

    def _update(self, iteration):
        Av = self._convolver.convolve(self._image_est)
        diff = Av - self._convolver.pad(self._image)
        grad = np.real(self._convolver.deconvolve(diff))
        self._image_est = self._image_est - self._step_size * grad
        self._image_est = np.maximum(self._image_est, 0)

    def _form_image(self):
        image = self._convolver.crop(self._image_est)
        image[image < 0] = 0
        return image
    
    def apply(self, image: np.ndarray, n_iter=10, disp_iter=10, ax=None, reset=True):
        self._image_est = (np.max(self._psf) + np.min(self._psf)) * self._convolver.pad(np.ones(self._psf.shape, dtype=np.float32)) / 2
        return super().apply(image, n_iter, disp_iter, ax, reset)