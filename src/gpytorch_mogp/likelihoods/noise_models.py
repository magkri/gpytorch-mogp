import warnings
from typing import Any, Optional

import torch
from linear_operator import to_linear_operator

from linear_operator.operators import ZeroLinearOperator, BlockDiagLinearOperator, BlockInterleavedLinearOperator
from torch import Tensor

# these imports would be relative if incorporated into gpytorch
from gpytorch import settings
from gpytorch.likelihoods.noise_models import Noise
from gpytorch.utils.warnings import NumericalWarning


class FixedMultiOutputGaussianNoise(Noise):
    def __init__(self, noise: Tensor, interleaved: bool = True) -> None:
        """
        :param noise: Should be an n x num_outputs x num_outputs tensor of noise covariance matrices per training point
        :param interleaved: If True, the noise is assumed to be block-diagonal w.r.t. inter-output covariances for each
            observation. If False, it is assumed to be block-diagonal w.r.t. inter-observation covariance for each output.
        """
        super().__init__()
        min_noise = settings.min_fixed_noise.value(noise.dtype)
        if noise.lt(min_noise).any():
            # check just the diagonal elements (the diagonal of each num_outputs x num_outputs matrix)
            if noise.diagonal(dim1=-2, dim2=-1).lt(min_noise).any():
                warnings.warn(
                    "Very small noise values detected along the diagonal. "
                    "This will likely lead to numerical instabilities. Rounding "
                    f"small noise values along diagonal up to {min_noise}.",
                    NumericalWarning,
                )
                # clamp just the diagonal elements
                noise = noise.diagonal(dim1=-2, dim2=-1).clamp_min(min_noise).diag_embed()
            else:
                warnings.warn(
                    "Very small noise values detected. That might be because "
                    "of zeros in the non-diagonal elements. Not doing anything "
                    "about it for now.",
                    NumericalWarning,
                )
                # noise = noise.clamp_min(min_noise)
        self.noise = to_linear_operator(noise)
        self.interleaved = interleaved

    def forward(
        self,
        *params: Any,
        shape: Optional[torch.Size] = None,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> BlockDiagLinearOperator | BlockInterleavedLinearOperator | ZeroLinearOperator:
        """
        - BlockDiagLinearOperator is used for interleaved noise.
        - BlockInterleavedLinearOperator is used for non-interleaved noise.
        """

        if shape is None:
            warnings.warn(
                f"I have not accounted for `shape is None`. Behavior may be unexpected."
                f"shape: {shape}"
            )
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]

        if noise is not None:
            if self.interleaved:
                return BlockDiagLinearOperator(to_linear_operator(noise), block_dim=-3)
            else:
                return BlockInterleavedLinearOperator(to_linear_operator(noise), block_dim=-3)
        elif shape[-1] == self.noise.shape[-1]:
            if self.interleaved:
                return BlockDiagLinearOperator(self.noise, block_dim=-3)
            else:
                return BlockInterleavedLinearOperator(self.noise, block_dim=-3)
        else:
            warnings.warn(
                "I have not accounted for this case. Behavior may be unexpected."
            )
            return ZeroLinearOperator()

    def _apply(self, fn):
        warnings.warn(
            "I have not accounted for use of _apply. Behavior may be unexpected."
        )
        self.noise = fn(self.noise)
        return super()._apply(fn)
