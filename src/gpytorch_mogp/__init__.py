import gpytorch

from gpytorch_mogp.likelihoods.multioutput_gaussian_likelihood import FixedNoiseMultiOutputGaussianLikelihood
from gpytorch_mogp.kernels.multioutput_kernel import MultiOutputKernel

# Monkey-patch gpytorch
# This allows users to do:
# >>> import gpytorch
# >>> import gpytorch_mogp
# >>> gpytorch.kernels.MultiOutputKernel
# <class 'gpytorch_mogp.multioutput_kernel.MultiOutputKernel'>
# >>> gpytorch.likelihoods.FixedNoiseMultiOutputGaussianLikelihood
# <class 'gpytorch_mogp.multioutput_gaussian_likelihood.FixedNoiseMultiOutputGaussianLikelihood'>
gpytorch.kernels.MultiOutputKernel = MultiOutputKernel
gpytorch.likelihoods.FixedNoiseMultiOutputGaussianLikelihood = FixedNoiseMultiOutputGaussianLikelihood

__all__ = ["MultiOutputKernel", "FixedNoiseMultiOutputGaussianLikelihood"]
