# gpytorch-mogp

[![PyPI - Version](https://img.shields.io/pypi/v/gpytorch-mogp.svg)](https://pypi.org/project/gpytorch-mogp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gpytorch-mogp.svg)](https://pypi.org/project/gpytorch-mogp)

`gpytorch-mogp` is a Python package that extends [GPyTorch](https://gpytorch.ai/) to support Multiple-Output Gaussian
processes where the correlation between the outputs is known.

> **Note:** The package is currently in an early stage of development. Expect bugs and breaking changes in future
> versions. Please create an issue if you encounter any problems or have any suggestions.

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Known Issues](#known-issues)
- [Glossary](#glossary)
- [TODO](#todo)
- [Contributing](#contributing)
- [License](#license)

## Installation

```shell
pip install gpytorch-mogp
```

> **Note:** If you want to use GPU acceleration, you should manually install
> [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support before running the above command.

If you want to run the [examples](examples), you should install the package from source instead. First, clone the
repository:

```shell
git clone https://github.com/magkri/gpytorch-mogp.git
```

Then install the package with the `examples` dependencies:

```shell
pip install .[examples]
```

To run the [comparison notebook](examples/comparison_with_rapid_models.ipynb), you also need to install
[`rapid-models`](https://github.com/RaPiD-models/rapid_models_dev2023) from source.

> **Note:** The version of `rapid-modles` linked to above is private at the time of writing, so you may not be able to
> install it.

## Usage

The package provides a custom `MultiOutputKernel` module that wraps one or more base kernels, producing a

```python
import gpytorch
from gpytorch_mogp import MultiOutputKernel, FixedNoiseMultiOutputGaussianLikelihood

# Define a multi-output GP model
class MultiOutputGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # Reuse the `MultitaskMean` module from `gpytorch`
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=2)
        # Use the custom `MultiOutputKernel` module
        self.covar_module = MultiOutputKernel(gpytorch.kernels.MaternKernel(), num_outputs=2)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Reuse the `MultitaskMultivariateNormal` distribution from `gpytorch`
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# Training data
train_x = ... # (n,) or (n, num_inputs)
train_y = ... # (n, num_outputs)
train_noise = ...  # (n, num_outputs, num_outputs)

# Initialize the model
likelihood = FixedNoiseMultiOutputGaussianLikelihood(noise=train_noise, num_tasks=2)
model = MultiOutputGPModel(train_x, train_y, likelihood)

# Training
model.train()
...

# Testing
model.eval()
test_x = ... # (m,) or (m, num_inputs)
test_noise = ...  # (m, num_outputs, num_outputs)
f_preds = model(test_x)
y_preds = likelihood(model(test_x), noise=test_noise)
```
> **Note:** `MultiOutputKernel` currently uses `num_outputs`, while `MultitaskMean` and
> `FixedNoiseMultiOutputGaussianLikelihood` use `num_tasks`. In this example, they mean the same thing. However, 
> multi-output and multi-task can have different meanings in other contexts. The reason for this inconsistency is that
> the `num_tasks` argument comes from `gpytorch`, while the `num_outputs` argument comes from `gpytorch-mogp`
> (`FixedNoiseMultiOutputGaussianLikelihood` inherits it from `gpytorch`). This inconsistency may be addressed in future
> versions of the package.

> **Note:** The `MultiOutputKernel` produces an [_interleaved_ block diagonal covariance matrix](#glossary) by default,
> as that is the convention used in `gpytorch`. If you want to produce a [_non-interleaved_ block diagonal covariance
> matrix](#glossary), you can pass `interleaved=False` to the `MultiOutputKernel` constructor.
>
> `MultitaskMultivariateNormal` expects an interleaved block diagonal covariance matrix by default. If you want to use a
> non-interleaved block diagonal covariance matrix, you can pass `interleaved=False` to the
> `MultitaskMultivariateNormal` constructor.
>
> `FixedNoiseMultiOutputGaussianLikelihood` expects the same noise structure regardless of interleaving, but will
> internally apply interleaving to the noise before adding it to the covariance matrix. Interleaving is applied by
> default. If you want to avoid this, you can pass `interleaved=False` to the `FixedNoiseMultiOutputGaussianLikelihood`
> constructor. The likelihood should always use the same interleaving setting as the kernel.
> 
> **WARNING:** The `interleaved=False` option is not working as expected at the moment. Avoid using it for now.

See also the [example notebooks/scripts](examples) for more usage examples.

See the [comparison notebook](examples/comparison_with_rapid_models.ipynb) for a comparison between the `gpytorch-mogp`
package and the `rapid-models` package (demonstrating that the two packages produce the same results for one example).

## Known Issues
- Constructing a GP with `interleaved=False` in `MultiOutputKernel`, `MultitaskMultivariateNormal`, and
    `FixedNoiseMultiOutputGaussianLikelihood` does not work as expected. Avoid using `interleaved=False` for now.
- Most issues are probably unknown at this point. Please create an issue if you encounter any problems ...

## Glossary
- **Multi-output GP model**: A Gaussian process model that produces multiple outputs for a given input. I.e. a model
    that predicts a vector-valued output for a given input.

- **Multi-output kernel**: A kernel that wraps one or more base kernels, producing a joint covariance matrix for the
    outputs. The joint covariance matrix is typically block diagonal, with each block corresponding to the output of a
    single base kernel. 

- **Block matrix**: A matrix that is partitioned into blocks. E.g. a block matrix with $2$ blocks per side of size $3$ 
    is structured like:
```math
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix} = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & b_{11} & b_{12} & b_{13} \\
a_{21} & a_{22} & a_{23} & b_{21} & b_{22} & b_{23} \\
a_{31} & a_{32} & a_{33} & b_{31} & b_{32} & b_{33} \\
c_{11} & c_{12} & c_{13} & d_{11} & d_{12} & d_{13} \\
c_{21} & c_{22} & c_{23} & d_{21} & d_{22} & d_{23} \\
c_{31} & c_{32} & c_{33} & d_{31} & d_{32} & d_{33}
\end{bmatrix}
```

- **Interleaved block matrix**: A block matrix where the blocks are interleaved. E.g. a block matrix with $2$ blocks per
    side of size $3$ is structured like:
```math
\begin{bmatrix}
a_{11} & b_{11} & a_{12} & b_{12} & a_{13} & b_{13} \\
c_{11} & d_{11} & c_{12} & d_{12} & c_{13} & d_{13} \\
a_{21} & b_{21} & a_{22} & b_{22} & a_{23} & b_{23} \\
c_{21} & d_{21} & c_{22} & d_{22} & c_{23} & d_{23} \\
a_{31} & b_{31} & a_{32} & b_{32} & a_{33} & b_{33} \\
c_{31} & d_{31} & c_{32} & d_{32} & c_{33} & d_{33}
\end{bmatrix}
```

- **(Non-interleaved) block diagonal covariance matrix**: A joint covariance matrix that is block diagonal, with each
    block corresponding to the output of a single base kernel. E.g. for a multi-output kernel with two base kernels,
    $`K_{\alpha}`$ and $`K_{\beta}`$, the joint covariance matrix is given by:
```math
K(\mathbf{X}, \mathbf{X}^*) = \begin{bmatrix}
K_{\alpha}(\mathbf{X}, \mathbf{X}^*) & \mathbf{0} \\
\mathbf{0} & K_{\beta}(\mathbf{X}, \mathbf{X}^*)
\end{bmatrix} = \begin{bmatrix}
K_{\alpha}(\mathbf{x}_1, \mathbf{x}_1^*) & \cdots & K_{\alpha}(\mathbf{x}_1, \mathbf{x}_n^*)\\
\vdots & \ddots & \vdots & & \mathbf{0} & \\
K_{\alpha}(\mathbf{x}_N, \mathbf{x}_1^*) & \cdots & K_{\alpha}(\mathbf{x}_N, \mathbf{x}_n^*) & & &\\
& & & K_{\beta}(\mathbf{x}_1, \mathbf{x}_1^*) & \cdots & K_{\beta}(\mathbf{x}_1, \mathbf{x}_n^*)\\
& \mathbf{0} & & \vdots & \ddots & \vdots\\
& & & K_{\beta}(\mathbf{x}_N, \mathbf{x}_1^*) & \cdots & K_{\beta}(\mathbf{x}_N, \mathbf{x}_n^*)
\end{bmatrix}
```

- **Interleaved block diagonal covariance matrix**: Similar to a block diagonal covariance matrix, but with the blocks
    interleaved:

```math
K(\mathbf{X}, \mathbf{X}^*) = \begin{bmatrix}
K_{\alpha}(\mathbf{x}_1, \mathbf{x}_1^*) & \mathbf{0} & \cdots & K_{\alpha}(\mathbf{x}_1, \mathbf{x}_n^*) & \mathbf{0} \\
\mathbf{0} & K_{\beta}(\mathbf{x}_1, \mathbf{x}_1^*) & \cdots & \mathbf{0} & K_{\beta}(\mathbf{x}_1, \mathbf{x}_n^*) \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
K_{\alpha}(\mathbf{x}_N, \mathbf{x}_1^*) & \mathbf{0} & \cdots & K_{\alpha}(\mathbf{x}_N, \mathbf{x}_n^*) & \mathbf{0} \\
\mathbf{0} & K_{\beta}(\mathbf{x}_N, \mathbf{x}_1^*) & \cdots & \mathbf{0} & K_{\beta}(\mathbf{x}_N, \mathbf{x}_n^*)
\end{bmatrix}
```

## Contributing

Contributions are welcome! Please create an issue or a pull request if you have any suggestions or improvements.

To get started contributing, clone the repository:

```shell
git clone https://github.com/magkri/gpytorch-mogp.git
```

Then install the package in editable mode with `all` dependencies:
```shell
pip install -e .[all]
```

## License

`gpytorch-mogp` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
