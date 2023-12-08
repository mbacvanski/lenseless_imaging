import torch

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
        self._PsiTPsi = torch.from_numpy(finite_diff_gram(self._padded_shape))
        self._R_divmat = 1.0 / (self._mu1 * (torch.abs(self._convolver._Hadj * self._convolver._H)) +
                                self._mu2 * torch.abs(self._PsiTPsi) + self._mu3)

    def reset(self):
        self._image_est = torch.zeros(tuple([1] + self._padded_shape))

        self._X = torch.zeros_like(self._image_est)
        self._U = torch.zeros_like(finite_diff(self._image_est))
        self._W = torch.zeros_like(self._X)
        self._forward_out = torch.zeros_like(self._X)
        self._Psi_out = torch.zeros_like(self._U)

        self._xi = torch.zeros_like(self._image_est)
        self._eta = torch.zeros_like(self._U)
        self._rho = torch.zeros_like(self._X)

        # precompute_X_divmat
        self._X_divmat = 1.0 / (self._convolver.pad(torch.ones(self._psf_shape)) + self._mu1)

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
        self._W = torch.maximum(self._rho / self._mu3 + self._image_est, torch.tensor(0))

    def _image_update(self):
        u = self._mu2 * self._U - self._eta
        rk = ((self._mu3 * self._W - self._rho)
              + finite_diff_adj(u)
              + self._convolver.deconvolve(self._mu1 * self._X - self._xi))

        # Compute the forward FFT on the last two dimensions
        freq_space_result = self._R_divmat * torch.fft.rfftn(rk, dim=(-3, -2))

        # Compute the inverse FFT to get the image estimate
        self._image_est = torch.fft.irfftn(freq_space_result, s=rk.shape[-3:-1], dim=(-3, -2))

    def _xi_update(self):
        # to avoid computing forward model twice
        self._xi += self._mu1 * (self._forward_out - self._X)

    def _eta_update(self):
        # to avoid finite difference operation again
        self._eta += self._mu2 * (self._Psi_out - self._U)

    def _rho_update(self):
        self._rho += self._mu3 * (self._image_est - self._W)

    def _update(self, iteration):
        # self._U_update()
        self._X_update()
        self._W_update()
        self._image_update()

        # update forward and sparse operators
        self._forward_out = self._convolver.convolve(self._image_est)
        self._Psi_out = finite_diff(self._image_est)

        self._xi_update()
        self._eta_update()
        self._rho_update()

    def _form_image(self):
        image = self._convolver.crop(self._image_est)
        image[image < 0] = 0
        return image

    def loss(self):
        """
        Compute the ADMM loss for image reconstruction.

        Parameters:
        image (torch.Tensor): The reconstructed image.
        observed_data (torch.Tensor): The observed data.
        psf (torch.Tensor): The point spread function (PSF) for convolution.

        Returns:
        torch.Tensor: The computed loss value.
        """
        # Ensure the image has non-negative values
        image = torch.clamp(self._image_est, min=0)

        # Data Fidelity Term: L2 norm of the difference between observed data and convolution of image with PSF
        # fidelity_loss = super().loss()
        forward_pass = self._convolver.convolve(self._image_est)
        fidelity_loss = torch.nn.functional.mse_loss(forward_pass, self._padded_image)

        image = image.squeeze(0)
        image = image.movedim(-1, -3)
        # print(image.shape)

        # Total Variation Regularization for each channel
        # kernel_x = torch.tensor([[[[-1, 1]]]], dtype=image.dtype).to(image.device)
        # kernel_y = torch.tensor([[[[-1], [1]]]], dtype=image.dtype).to(image.device)
        #
        # tv_loss = 0
        # for channel in range(image.shape[1]):
        #     channel_img = image[:, channel:channel + 1, :, :]
        #     grad_x = torch.abs(torch.nn.functional.conv2d(channel_img, kernel_x, padding='same'))
        #     grad_y = torch.abs(torch.nn.functional.conv2d(channel_img, kernel_y, padding='same'))
        #     tv_loss += self._tau * torch.sum(grad_x + grad_y)

        grad_x = image[:, :, :, :-1] - image[:, :, :, 1:]
        grad_y = image[:, :, :-1, :] - image[:, :, 1:, :]
        tv_loss = self._tau * (torch.sum(torch.abs(grad_x)) + torch.sum(torch.abs(grad_y)))

        # Non-negativity constraint is implicitly handled by clamping the image to non-negative values

        # Total Loss
        total_loss = fidelity_loss + tv_loss

        return total_loss
