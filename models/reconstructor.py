import abc

import numpy as np
from matplotlib import pyplot as plt

from utils import RealFFTConvolve2D, plot_image


class ReconstructionAlgorithm(abc.ABC):
    """
    Abstract class for defining lensless imaging reconstruction algorithms.

    The following abstract methods need to be defined:

    * :py:class:`~lensless.ReconstructionAlgorithm._update`: updating state variables at each iteration.
    * :py:class:`~lensless.ReconstructionAlgorithm.reset`: reset state variables.
    * :py:class:`~lensless.ReconstructionAlgorithm._form_image`: any pre-processing that needs to be done in order to view the image estimate, e.g. reshaping or clipping.

    Functionality for iterating, saving, and visualization is already implemented in the `apply` method.

    Using a reconstruction algorithm that derives from it involves three steps:
    #. Creating an instance of the reconstruction algorithm.
    #. Setting the data.
    #. Applying the algorithm.
    """

    def __init__(self, psf, **kwargs):
        """
        Base constructor. Derived constructor may define new state variables
        here and also reset them in `reset`.

        Parameters
        ----------
            psf : :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
                Point spread function (PSF) that models forward propagation.
                Must be of shape (depth, height, width, channels) even if
                depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
                to load a PSF from a file such that it is in the correct format.
            pad : bool, optional
                Whether data needs to be padded prior to convolution. User may wish to
                optimize padded data and set this to False, as is done for :py:class:`~lensless.ADMM`.
                Defaults to True.
            n_iter : int, optional
                Number of iterations to run algorithm for. Can be overridden in `apply`.
            reset : bool, optional
                Whether to reset state variables in the base constructor. Defaults to True.
                If False, you should call reset() at one point to initialize state variables.
        """
        super().__init__()
        assert len(psf.shape) == 4, "PSF must be 4D: (depth, height, width, channels)."
        assert psf.shape[3] == 3 or psf.shape[3] == 1, "PSF must either be rgb (3) or grayscale (1)"

        self._psf_shape = np.array(psf.shape)
        self._psf = psf.astype(np.float32)

        self._image = None

        self.reset()

    @abc.abstractmethod
    def reset(self):
        """
        Reset state variables.
        """
        ...

    @abc.abstractmethod
    def _update(self, iteration):
        """
        Update state variables.
        """
        ...

    @abc.abstractmethod
    def _form_image(self):
        """
        Any pre-processing to form a viewable image, e.g. reshaping or clipping.
        """
        ...

    def _progress(self):
        """
        Optional method for printing progress update, e.g. relative improvement
        in reconstruction.
        """
        ...

    def apply(self, image: np.ndarray, n_iter=10, disp_iter=10, ax=None, reset=True):
        """
        Method for performing iterative reconstruction. Note that `set_data`
        must be called beforehand.

        Parameters
        ----------
        image: :py:class:`~numpy.ndarray`
            Lensless data on which to iterate to recover an estimate of the scene.
        n_iter : int, optional
            Number of iterations. If not provided, default to `self._n_iter`.
        disp_iter : int, optional
            How often to display and/or intermediate reconstruction (in number
            of iterations). If `None` OR `plot` or `save` are False, no
            intermediate reconstruction will be plotted/saved.
        ax : :py:class:`~matplotlib.axes.Axes`, optional
            `Axes` object to fill for plotting/saving, default is to create one.
        reset : bool, optional
            Whether to reset state variables before applying reconstruction. Default to True.
            Set to false if continuing reconstruction from previous state.

        Returns
        -------
        final_im : :py:class:`~numpy.ndarray`
            Final reconstruction.
        ax : :py:class:`~matplotlib.axes.Axes`
            `Axes` object on which final reconstruction is displayed. Only
            returning if `plot` or `save` is True.

        """
        assert isinstance(image, np.ndarray)
        assert len(image.shape) >= 3, "Data must be at least 3D: [..., width, height, channel]."
        assert np.all(self._psf_shape[-3:-1] == np.array(image.shape)[-3:-1]), "PSF and data shape mismatch"

        self._image = image[None, None, ...]

        if reset:
            self.reset()

        if disp_iter is not None:
            if ax is None:
                img = self._form_image()
                plot_image(img[0])

        for i in range(n_iter):
            self._update(i)

            if disp_iter is not None and (i + 1) % disp_iter == 0:
                self._progress()
                img = self._form_image()
                data1 = img[0]
                ax = plot_image(data1)
                ax.set_title("Reconstruction after iteration {}".format(i + 1))
                plt.draw()

        final_im = self._form_image()[0]
        ax = plot_image(final_im)
        ax.set_title("Final reconstruction after {} iterations".format(n_iter))
        return final_im, ax
