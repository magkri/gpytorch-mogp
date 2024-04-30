import warnings
from typing import Any, Optional

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.likelihoods import _MultitaskGaussianLikelihoodBase
from gpytorch.utils.warnings import GPInputWarning
from linear_operator.operators import (
    LinearOperator,
    ZeroLinearOperator,
)
from torch import Tensor
from torch.distributions import Normal

from .noise_models import FixedMultiOutputGaussianNoise


class FixedNoiseMultiOutputGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    def __init__(
        self,
        noise: Tensor,
        interleaved: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            noise_covar=FixedMultiOutputGaussianNoise(noise=noise, interleaved=interleaved), *args, **kwargs
        )

    def _shaped_noise_covar(
        self,
        base_shape: torch.Size,
        *params: Any,
        **kwargs: Any,
    ) -> LinearOperator:

        res = self.noise_covar(*params, shape=base_shape, **kwargs)

        if isinstance(res, ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseMultiOutputGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        # interleaving already handled in FixedMultiOutputGaussianNoise, this is different from the way it's done in
        # MultitaskGaussianLikelihood. There they handle it by including an `interleaved` kwarg in this method and
        # using an if-statement here to switch the order of the operands passed to the Kronecker product.

        return res

    def marginal(
        self, function_dist: MultitaskMultivariateNormal, *params: Any, **kwargs: Any
    ) -> MultitaskMultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()

        noise_covar = self._shaped_noise_covar(
            # TODO: should params be left out here? It contains values sometimes, they might be handled incorrectly
            mean.shape, *params, **kwargs
        )
        covar = covar + noise_covar

        return function_dist.__class__(mean, covar, interleaved=function_dist._interleaved)

    def forward(self, *args, **kwargs):
        # this is for safety, as the inherited method calls .diagonal() on the output of _shaped_noise_covar()
        raise NotImplementedError("Forward method not implemented for FixedNoiseMultiOutputGaussianLikelihood")

    def expected_log_prob(self, *args, **kwargs):
        # this is for safety, as the inherited method calls .diagonal() on the output of _shaped_noise_covar()
        raise NotImplementedError("Expected log probability method not implemented for FixedNoiseMultiOutputGaussianLikelihood")

    @property
    def has_global_noise(self) -> bool:
        warnings.warn("`has_global_noise` accessed. Behavior may be unexpected.")
        return False

    @property
    def has_task_noise(self) -> bool:
        warnings.warn("`has_task_noise` accessed. Behavior may be unexpected.")
        return False
