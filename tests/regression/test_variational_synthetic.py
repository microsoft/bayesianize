import pytest

import torch
import torch.distributions as dist


import models.bnn as bnn


def bayesian_regression(n, d, data_precision, prior_precision):
    x = torch.randn(n, d)
    w = torch.randn(d, 1) * prior_precision ** -0.5
    y = x.mm(w) + torch.randn(n, 1) * data_precision ** -0.5
    posterior_precision = data_precision * x.t().mm(x) + prior_precision * torch.eye(d)
    posterior_mean = data_precision * torch.solve(x.t().mm(y), posterior_precision).solution

    return x, y, posterior_mean, posterior_precision


def fit_(module, x, y, data_precision, lr, epochs, num_decays):
    optim = torch.optim.Adam(module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, epochs // num_decays, 0.1)
    for _ in range(epochs):
        optim.zero_grad()
        yhat = module(x)
        nll = -dist.Normal(yhat, data_precision ** -0.5).log_prob(y).sum()
        kl = module.kl_divergence()
        loss = nll + kl
        loss.backward()
        optim.step()
        scheduler.step()

@pytest.mark.parametrize("n,d,data_precision,prior_precision,local_reparameterization", [
    (10, 3, 100., 1., True),
    (10, 3, 100., 1., False),
    (100, 6, 10, 5., True)
])
def test_ffg(n, d, data_precision, prior_precision, local_reparameterization):
    torch.manual_seed(42)
    x, y, mu, lamda = bayesian_regression(n, d, data_precision, prior_precision)

    l = bnn.nn.FFGLinear(d, 1, bias=False, prior_weight_sd=prior_precision ** -0.5)
    if not local_reparameterization:
        l.eval()
    fit_(l, x, y, data_precision, 1e-1, 4000, 4)

    assert torch.allclose(mu.squeeze(), l.weight_mean.squeeze(), atol=1e-2)
    assert torch.allclose(lamda.diagonal() ** -0.5, l.weight_sd.squeeze(), atol=1e-2)
    assert dist.kl_divergence(dist.Normal(l.weight_mean.squeeze(), l.weight_sd.squeeze()),
                              dist.Normal(mu.squeeze(), lamda.diag() ** -0.5)).sum().item() < 1e-2


@pytest.mark.parametrize("n,d,data_precision,prior_precision", [
    (10, 3, 100., 1.),
    (25, 4, 250., 100.)
])
def test_fcg(n, d, data_precision, prior_precision):
    torch.manual_seed(42)
    x, y, mu, lamda = bayesian_regression(n, d, data_precision, prior_precision)

    l = bnn.nn.FCGLinear(d, 1, bias=False, prior_weight_sd=prior_precision ** -0.5)
    fit_(l, x, y, data_precision, 0.1, 4000, 4)

    assert torch.allclose(mu.squeeze(), l.mean, atol=1e-2)
    assert torch.allclose(lamda.inverse(), l.scale_tril.mm(l.scale_tril.t()), atol=1e-2)
    assert dist.kl_divergence(l.parameter_distribution,
                              dist.MultivariateNormal(mu.squeeze(), precision_matrix=lamda)).item() < 1e-2
