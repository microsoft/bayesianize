# bayesianize: A Bayesian neural network wrapper in pytorch

This is a wrapper that converts a deterministic neural network to a Bayesian neural network.

Example usage: converting a deterministic ResNet18 network to a Bayesian neural network version of it, with mean-field variational inference by default.
```
from bnn import bayesianize_
import torchvision.models as models
net = models.resnet18()
bayesian_net = bayesianize_(net)
```
The wrapper currently supports converting pytorch networks with ``torch.nn.Linear'', ``torch.nn.Conv1d'', ``torch.nn.Conv2d'', ``torch.nn.Conv3d'' modules only. These include fully connected neural networks, convolutional neural networks and Resnets.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
