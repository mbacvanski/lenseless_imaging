# Getting Data

A 200 image subset of the X dataset can be found here, which includes the PSF file, lensed, and diffuser camera files.
For some reason only the `.npy` files are working at the moment.

https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE

# Organization

Different models go in the `algorithms` folder, and inherit from the `ReconstructionAlgorithm` abstract class. The
abstract class provides a method `apply(image, <options>)` which calls `self._update()` repeatedly.

To implement a new reconstruction algorithm (see `algorithms/admm.py` for reference):

1. Subclass `ReconstructionAlgorithm`
2. Implement `reset(self)` that defines whatever variables you need
3. Implement `_update(self)` that performs one iteration of the algorithm
4. Implement `_form_image(self)` that returns the current image during the algorithm's iteration
