import numpy as np

from algorithms.gradient_descent import gradient_descent
from utils import RealFFTConvolve2D


# noinspection PyPep8Naming
class nesterov_gradient_descent(gradient_descent):
    """
    Object for applying gradient descent with nestrov momentum
    """

    def __init__(self, psf, gt=None, p=0, mu =0.9, norm="backward"):
        """
        Parameters
        ----------
        psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        p : float
            Momentum tracker parameter. Default is 0.
        mu : float
            Momentum parameter. Default is 0.9.
        
        """
        self._p = p
        self._mu = mu
        super(nesterov_gradient_descent, self).__init__(psf=psf, gt=gt, norm=norm)

    def reset(self, p=0, mu=0.9):
        self._p = p
        self._mu = mu
        super(nesterov_gradient_descent, self).reset()

    def _update(self, iteration):
        last_p = self._p
        self._p = self._mu * self._p - self._step_size * self._grad()
        diff = -self._mu * last_p + (1 + self._mu) * self._p
        self._image_est = self._image_est + diff
        self._image_est = np.maximum(self._image_est, 0)