"""Based on: https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html."""

import math

import gpytorch
import torch
from matplotlib import pyplot as plt

from gpytorch_mogp.kernels import MultiOutputKernel
from gpytorch_mogp.likelihoods import FixedNoiseMultiOutputGaussianLikelihood

# config
seed = 3407
interleaved = True
train = True

# seed
torch.manual_seed(seed)

train_x = torch.linspace(0, 1, 100)
train_y = torch.stack(
    [
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    ],
    -1,
)
train_sigma = torch.eye(2).expand(train_x.size(0), 2, 2) * 0.2


class MultiOutputGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=2)
        self.covar_module = MultiOutputKernel(
            gpytorch.kernels.MaternKernel(),
            num_outputs=2,
            make_copies=True,
            interleaved=interleaved,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved=interleaved)


likelihood = FixedNoiseMultiOutputGaussianLikelihood(noise=train_sigma, interleaved=interleaved, num_tasks=2)
model = MultiOutputGPModel(train_x, train_y, likelihood)


if train:
    training_iterations = 100

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
        optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 100)
    f_preds = model(test_x)
    predictions = likelihood(f_preds, noise=torch.eye(2).expand(test_x.size(0), 2, 2) * 0.2)
    samples = predictions.sample(torch.Size((100,))).numpy()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

mean = predictions.mean
lower, upper = predictions.confidence_region()

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task

# Plot training data as black stars
y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), "k*")
# Predictive mean as blue line
y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), "b")
# Shade in confidence
y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
y1_ax.set_ylim([-3, 3])
y1_ax.legend(["Observed Data", "Mean", "Confidence"])
y1_ax.set_title("Observed Values (Likelihood)")

# Plot training data as black stars
y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), "k*")
# Predictive mean as blue line
y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), "b")
# Shade in confidence
y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
y2_ax.set_ylim([-3, 3])
y2_ax.legend(["Observed Data", "Mean", "Confidence"])
y2_ax.set_title("Observed Values (Likelihood)")
plt.show()
