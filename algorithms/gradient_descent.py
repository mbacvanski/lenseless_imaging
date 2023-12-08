import numpy as np
import torch

from algorithms.reconstruction_algorithm import ReconstructionAlgorithm
from utils import RealFFTConvolve2D


# noinspection PyPep8Naming
class gradient_descent(ReconstructionAlgorithm):
    """
    Object for applying conventional gradient descent
    """

    def __init__(self, psf, gt=None, norm="backward"):
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
        super(gradient_descent, self).__init__(psf=psf, gt=gt, norm=norm)

    def reset(self):
        # Assuming self._padded_shape is a tuple representing the shape
        self._image_est = torch.zeros(self._padded_shape, dtype=torch.float32)
        # self._image_est = torch.rand(self._padded_shape, dtype=torch.float32)
        # self._image_est = torch.ones(self._padded_shape, dtype=torch.float32) * 127

        step_size_numerator = torch.abs(self._convolver._H * self._convolver._Hadj)
        self._step_size = 1.8 / torch.max(step_size_numerator).real
        # self._step_size = .0005
        self._losses_history = []
        self._grad_history = []
        self._computed_grad_history = []

    def _grad(self):
        """
        Calculate the gradient of the loss with respect to the image estimate.
        """
        if self._image_est.grad is not None:
            self._image_est = self._image_est.clone()
            self._image_est.grad.zero_()
            self._image_est.requires_grad = True

        forward_pass = self._convolver.convolve(self._image_est)
        loss = torch.nn.functional.mse_loss(forward_pass, self._padded_image)
        self._losses_history.append(loss.item())

        loss.backward()
        grad = self._image_est.grad
        self._image_est = self._image_est.detach()
        return grad  # derivative of loss with respect to self._image_est

    def _update(self, iteration):
        grad = self._grad()  # Call the new grad method
        self._grad_history.append(grad.detach().numpy())

        with torch.no_grad():  # disable gradient tracking
            self._image_est -= self._step_size * grad * 1e6
            self._image_est = torch.clamp(self._image_est, min=0, max=1)

    def _form_image(self):
        image = self._convolver.crop(self._image_est)
        image = torch.clamp(image, min=0)
        return image.detach()
