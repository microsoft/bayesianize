import copy
import itertools

import pytest

import torch
import torch.nn as nn


from models.bnn import bayesianize_
from models.bnn.nn import InducingLinear, InducingConv2d, InducingDeterministicLinear, InducingDeterministicConv2d
    

@pytest.mark.parametrize("q_inducing,whitened,max_lamda,max_sd_u,bias,layer_type,sqrt_width_scaling", itertools.product(
    ("diagonal", "matrix", "full"),
    (False, True),
    (None, 0.3),
    (None, 0.3),
    (False, True),
    ("linear", "conv"),
    (False, True)
))
def test_forward_shape(q_inducing, whitened, max_lamda, max_sd_u, bias, layer_type, sqrt_width_scaling):
    inducing_rows = 4
    inducing_cols = 2
    batch_size = 5
    inducing_kwargs = dict(
        inducing_rows=inducing_rows, inducing_cols=inducing_cols, q_inducing=q_inducing, whitened_u=whitened,
        max_lamda=max_lamda, max_sd_u=max_sd_u, init_lamda=1, bias=bias, sqrt_width_scaling=sqrt_width_scaling)

    if layer_type == "linear":
        in_features = 3
        out_features = 5
        layer = InducingLinear(in_features, out_features, **inducing_kwargs)

        x = torch.randn(batch_size, in_features)
        expected_shape = (batch_size, out_features)
    elif layer_type == "conv":
        in_channels = 3
        out_channels = 6
        kernel_size = 3
        padding = 1
        layer = InducingConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, **inducing_kwargs)

        h, w = 7, 7
        x = torch.randn(batch_size, in_channels, h, w)
        expected_shape = (batch_size, out_channels, h, w)
    else:
        raise ValueError(f"Invalid layer_type: {layer_type}")

    assert layer(x).shape == expected_shape


@pytest.mark.parametrize("inference", ["inducing", "inducingdeterministic"])
def test_bayesianize_compatible(inference):
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Conv2d(8, 8, 3), nn.Linear(32, 16), nn.Linear(16, 8))
    bnn = copy.deepcopy(net)
    bayesianize_(bnn, inference)

    for m, bm in zip(net.modules(), bnn.modules()):
        if m is net:
            continue

        if inference == "inducing":
            if isinstance(m, nn.Linear):
                assert isinstance(bm, InducingLinear)
            elif isinstance(m, nn.Conv2d):
                assert isinstance(bm, InducingConv2d)
            else:  # unreachable
                assert False
        else:
            if isinstance(m, nn.Linear):
                assert isinstance(bm, InducingDeterministicLinear)
            elif isinstance(m, nn.Conv2d):
                assert isinstance(bm, InducingDeterministicConv2d)
            else:  # unreachable
                assert False
