import numpy as np

from algorithms.gradient_descent import gradient_descent
from utils import RealFFTConvolve2D


# noinspection PyPep8Naming
class FISTA(gradient_descent):
    """
    Object for applying gradient descent with nestrov momentum
    """

    def __init__(self, psf, gt=None, tk=1.0, norm="backward"):
        """
        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        tk : float
            Initialize step size with this. Default is 1.0.
        
        """
        self._init_tk = tk
        self._tk = tk
        super(FISTA, self).__init__(psf, gt=gt, norm=norm)
        self._xk = self._image_est

    def reset(self, p=0, mu=0.9):
        self._tk = self._init_tk
        super(FISTA, self).reset()
        self._xk = self._image_est

    def _update(self, iteration):
        self._image_est = self._image_est - self._step_size * self._grad()
        xk = np.maximum(self._image_est, 0)
        tk = (1 + np.sqrt(1 + 4 * self._tk**2)) / 2
        self._image_est = xk + (self._tk - 1) / tk * (xk - self._xk)
        self._xk = xk
        self._tk = tk