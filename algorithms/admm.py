import numpy as np
from scipy import fft

from algorithms.reconstruction_algorithm import ReconstructionAlgorithm
from utils import finite_diff_gram, finite_diff, finite_diff_adj, soft_thresh, RealFFTConvolve2D


# noinspection PyPep8Naming
class ADMM(ReconstructionAlgorithm):
    """
    Object for applying ADMM (Alternating Direction Method of Multipliers) with
    a non-negativity constraint and a total variation (TV) prior.

    Paper about ADMM: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    Slides about ADMM: https://web.stanford.edu/class/ee364b/lectures/admm_slides.pdf
    """

    def __init__(self, psf, gt=None, mu1=1e-6, mu2=1e-5, mu3=4e-5, tau=0.0001, norm="backward"):
        """
        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        mu1 : float
            Step size for updating primal/dual variables.
        mu2 : float
            Step size for updating primal/dual variables.
        mu3 : float
            Step size for updating primal/dual variables.
        tau : float
            Weight for L1 norm of `psi` applied to the image estimate.
        pad : bool
            Whether to pad the image with zeros before applying the PSF. Default
            is False, as optimized data is already padded.
        norm : str
            Normalization to use for the convolution. Options are "forward",
            "backward", and "ortho". Default is "backward".
        """
        self._mu1 = mu1
        self._mu2 = mu2
        self._mu3 = mu3
        self._tau = tau

        self._convolver = RealFFTConvolve2D(psf, norm=norm)
        self._padded_shape = self._convolver._padded_shape

        # 3D ADMM is not supported yet
        assert len(psf.shape) == 4, "PSF must be 4D: (depth, height, width, channels)."

        # call reset() to initialize matrices
        super(ADMM, self).__init__(psf=psf, gt=gt, norm=norm)

        # set prior
        self._PsiTPsi = finite_diff_gram(self._padded_shape)
        self._R_divmat = 1.0 / (self._mu1 * (np.abs(self._convolver._Hadj * self._convolver._H)) +
                                self._mu2 * np.abs(self._PsiTPsi) + self._mu3).astype(np.complex64)

    def reset(self):
        self._image_est = np.zeros([1] + self._padded_shape, dtype=np.float32)

        self._X = np.zeros_like(self._image_est)
        self._U = np.zeros_like(finite_diff(self._image_est))
        self._W = np.zeros_like(self._X)
        self._forward_out = np.zeros_like(self._X)
        self._Psi_out = np.zeros_like(self._U)

        self._xi = np.zeros_like(self._image_est)
        self._eta = np.zeros_like(self._U)
        self._rho = np.zeros_like(self._X)

        # precompute_X_divmat
        self._X_divmat = 1.0 / (self._convolver.pad(np.ones(self._psf_shape, dtype=np.float32)) + self._mu1)

    def _U_update(self):
        """Total variation update."""
        # to avoid computing sparse operator twice
        self._U = soft_thresh(self._Psi_out + self._eta / self._mu2, self._tau / self._mu2)

    def _X_update(self):
        # to avoid computing forward model twice
        # self._X = self._X_divmat * (self._xi + self._mu1 * self._forward_out + self._data)
        self._X = self._X_divmat * (self._xi + self._mu1 * self._forward_out + self._convolver.pad(self._image))

    def _W_update(self):
        """Non-negativity update"""
        self._W = np.maximum(self._rho / self._mu3 + self._image_est, 0)

    def _image_update(self):
        u = self._mu2 * self._U - self._eta
        rk = ((self._mu3 * self._W - self._rho) + finite_diff_adj(u) + self._convolver.deconvolve(self._mu1 * self._X - self._xi))

        freq_space_result = self._R_divmat * fft.rfft2(rk, axes=(-3, -2))
        self._image_est = fft.irfft2(freq_space_result, axes=(-3, -2))

    def _xi_update(self):
        # to avoid computing forward model twice
        self._xi += self._mu1 * (self._forward_out - self._X)

    def _eta_update(self):
        # to avoid finite difference operation again
        self._eta += self._mu2 * (self._Psi_out - self._U)

    def _rho_update(self):
        self._rho += self._mu3 * (self._image_est - self._W)

    def _update(self, iteration):
        self._U_update()
        self._X_update()
        self._W_update()
        self._image_update()

        # update forward and sparse operators
        self._forward_out = self._convolver.convolve(self._image_est).numpy()
        self._Psi_out = finite_diff(self._image_est)

        self._xi_update()
        self._eta_update()
        self._rho_update()

    def _form_image(self):
        image = self._convolver.crop(self._image_est)
        image[image < 0] = 0
        return image
