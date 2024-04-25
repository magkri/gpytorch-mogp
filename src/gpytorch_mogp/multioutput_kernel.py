from copy import deepcopy

from gpytorch.kernels import Kernel
from linear_operator import to_linear_operator
from linear_operator.operators import (
    BlockDiagLinearOperator,
    BlockInterleavedLinearOperator,
)
from linear_operator.operators import cat as linear_operator_cat
from torch.nn import ModuleList


class MultiOutputKernel(Kernel):
    def __init__(
        self,
        *covar_modules: Kernel,
        num_outputs: int | None = None,
        make_copies: bool = True,
        interleaved: bool = True,
        **kwargs,
    ):
        """Initialize a multi-output kernel.

        Args:
        ----
            *covar_modules: Variable number of covariance kernels. If a single kernel is provided, it is either replicated
                across outputs (if make_copies=True) or shared among all (if make_copies=False). If multiple kernels are given,
                each kernel corresponds to one output dimension.
            num_outputs: Optional; The number of output dimensions. Required if a single kernel is provided. If provided
                with multiple kernels, it must match the number of kernels.
            make_copies: Determines whether to replicate the kernel for each output dimension (True) or
                to share the same instance across all (False). Ignored if multiple kernels are provided.
            interleaved: If True, the covariance matrix is block-diagonal w.r.t. inter-output covariances for each
                observation. If False, it is block-diagonal w.r.t. inter-observation covariance for each output.
            **kwargs: Additional keyword arguments for the Kernel base class.

        """
        super().__init__(**kwargs)

        if len(covar_modules) == 1 and num_outputs is None:
            raise RuntimeError(
                "`num_outputs` must be specified if a single kernel is provided"
            )

        if (
            len(covar_modules) > 1
            and num_outputs is not None
            and len(covar_modules) != num_outputs
        ):
            raise RuntimeError(
                "`num_outputs` must match the number of kernels provided"
            )

        # if not isinstance(covar_modules, list) or (len(covar_modules) != 1 and len(covar_modules) != num_outputs):
        #     raise RuntimeError("`covar_modules` should be a list of kernels of length either 1 or num_outputs")

        if len(covar_modules) == 1:
            # Copy the single kernel for each output dimension if make_copies=True, otherwise use the same instance
            # TODO: don't make a copy of the first kernel, just use it as is
            # covar_modules = [deepcopy(covar_modules[0]) for _ in range(num_outputs)] if make_copies else covar_modules
            covar_modules = (
                [
                    deepcopy(covar_modules[0]) if i > 0 else covar_modules[0]
                    for i in range(num_outputs)
                ]
                if make_copies
                else covar_modules
            )
            # covar_modules = covar_modules + [deepcopy(covar_modules[0]) for i in range(num_outputs - 1)] if make_copies else covar_modules

        self.covar_modules = ModuleList(covar_modules)
        self.num_outputs = (
            num_outputs if num_outputs is not None else len(covar_modules)
        )
        self.interleaved = interleaved

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError(
                "MultiOutputKernel does not accept the last_dim_is_batch argument."
            )

        # Forward all the covar_modules
        output_covars = [
            module.forward(x1, x2, **params) for module in self.covar_modules
        ]

        if len(self.covar_modules) == 1 and self.num_outputs > 1:
            # Sharing the same kernel across all outputs, so we need to expand the output
            output_covars = output_covars[0].expand(self.num_outputs, *output_covars[0].shape)
        else:
            # Combine all outputs into a single LinearOperator
            # Would prefer to use torch.stack here, but it is not supported by linear_operator, so we unsqueeze and cat
            # (torch.cat does not work atm., have to use the one from linear_operator instead)
            output_covars = linear_operator_cat(
                [oc.unsqueeze(0) for oc in output_covars]
            )  # returns a regular Tensor if all inputs are regular Tensors

        # Make sure we have a LinearOperator
        output_covars = to_linear_operator(output_covars)

        # temp: must convert to dense linear operator here (or Tensor) as indexing a (lazy) block linear operator seems
        #   to be bugged atm. (I (magnus.kristiansen@dnv.com) have documented this as an issue, somewhere...).
        if self.interleaved:
            res = to_linear_operator(BlockInterleavedLinearOperator(output_covars, block_dim=-3).to_dense())
        else:
            res = to_linear_operator(BlockDiagLinearOperator(output_covars, block_dim=-3).to_dense())

        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def num_outputs_per_input(self, x1, x2):
        """Determines the number of outputs per input for the multi-output kernel."""
        return self.num_outputs
