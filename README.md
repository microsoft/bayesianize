# Bayesianize: a Bayesian neural network wrapper in pytorch

Bayesianize is a lightweight Bayesian neural network (BNN) wrapper in pytorch. The overall goal is to allow for easy conversion of neural networks in existing scripts to BNNs with minimal changes to the code. 

Currently the wrapper supports the following uncertainty estimation methods for feed-forward neural networks and convnets:

* Mean-field variational inference (MFVI): variational inference with fully factorised Gaussian (FFG) approximation.
* Variational inference with full-covariance Gaussian approximation (for each layer).
* Variational inference with inducing weights: each of the layer is augmented with a small matrix of inducing weights, then MFVI is performed in the inducing weight space.
* Ensemble in inducing weight space: same augmentation as above, but with ensembles in the inducing weight space.

## Usage

The main workhorse of our library is the `bayesianize_` function.
It can be applied to a pytorch neural network and turns deterministic `nn.Linear` and `nn.Conv` layers into their bayesian counterparts.
For example, to construct a Bayesian ResNet-18 that uses the variational inducing weight method, run:
```
import bnn
net = torchvision.models.resnet18()
bnn.bayesianize_(net, inference="inducing", inducing_rows=64, inducing_cols=64)
```

Then the converted BNN can be trained in almost identical way as one would train a deterministic net:
```
yhat = net(x_train)
nll = F.cross_entropy(yhat, y_train)
kl = sum(m.kl_divergence() for m in net.modules()
         if hasattr(m, "kl_divergence"))
loss = nll + kl / dataset_size
loss.backward()
optim.step()
```
The main difference to training a deterministic net is the extra KL-divergence regulariser in the objective function.
Note that while the call to the forward method of the net looks the same, it is no longer deterministic because the weights are sampled, so to subsequent calls will lead to different predictions.
Therefore, when testing, an average of multiple predictions is needed. For example, in BNN classification:
```
net.eval()
with torch.no_grad():
    logits = torch.stack([net(x_test) for _ in range(num_samples)])
    probs = logits.softmax(-1).mean(0)
```

Besides the inducing weight method, other variational inference approaches can be used, by setting `inference="ffg"` for MFVI or `inference="fcg"` for VI with full-covariance Gaussians.

`bayesianize_` also supports using different methods or arguments for different layers, by passing in a dictionary for the `inference` argument.
This way you can, for example, take a pre-trained ResNet and only perform (approximate) Bayesian inference over the weights of the final, linear layer (which you can access via the `net.fc` attribute):
```
bnn.bayesianize_(net, inference={
    net.fc: {"inference": "fcg"}
})
optim = torch.optim.Adam(net.fc.parameters(), 1e-3)
```
If `net` is an instance of `nn.Sequential` the network layers can also be indexed as `net[i]`, e.g. `net[-1]` for the output layer.
Alternatively, it is possible use the names of layers (e.g. `"fc"` for the linear output layer of a ResNet), the names of classes (`"Linear"`) or the corresponding objects as keys for the dictionary to specify the inference arguments for individual or groups of layers.

## Installation

The easiest option for installing the library is to first create a `bayesianize` conda environment from the `.yml` file we provide:
```
conda env create -f environment.yml
```
Depending on your system, you might need to add a `cudatoolkit` or `cpuonly` as the final line to the `environment.yml` to install the correct version of pytorch, e.g. add
```
  - cudatoolkit=11.0
```
to install pytorch with CUDA11 support.

Then you can load the environment and pip install our module from source:
```
conda activate bayesianize
pip install -e .
```

Alternatively, you can copy the `bnn/` folder to your project or add  `/your_path/bnn/` to your `PYTHONPATH`:
```
export PYTHONPATH=PATH_TO_INDUCING_WEIGHT_DIR:$PYTHONPATH
```
with `PATH_TO_INDUCING_WEIGHT_DIR=/your_path/` in the example case.

## Code structure

The variational inference logic is mostly contained inside the `bnn.nn.mixins.variational` module.
There we implement mixin classes that contain logic for sampling `.weight` and `.bias` parameters from a variational posterior and calculating its KL divergence from a prior.
Those classes are mixed with pytorch's `nn.Linear` and all `nn.Conv` classes in `bnn/nn/modules.py`.
Our `bayesianize_` method automatically collects classes that inherit from `bnn.nn.mixins.base.BayesianMixin` and the Linear or Conv class.
So if you want to add your own variational layer classes, e.g. with a low rank or matrix normal variational posterior, you only need to make them inherit from our `BayesianMixin` class and create the corresponding linear and conv layers in `modules`.

## Example script

We provide an example script for training Bayesian ResNets on CIFAR10 and CIFAR100 in `scripts/cifar_resnet.py`.
The most important command line argument is the `--inference-config`.
If you do not provide a value, your network will remain unchanged and the script will train using maximum likelihood.
Otherwise you can pass in the path to one of the inference config files in the `configs/` directory.
We provide configs for Gaussian mean-field VI with either no contraints on the variational posterior or one where the maximum standard deviation is set to 0.1.
There are also of course configs for our inducing weight method, both with an ensemble and fully-factorised Gaussian VI in inducing space.
Note that there are separate configs for CIFAR10 and CIFAR100 due to the different number of classes. 

To train a BNN with our inducing weight method, you can run the script for example as:
```
python scripts/cifar_resnet.py --inference-config=configs/ffg_u_cifar10.json \
    --num-epochs=200 --ml-epochs=100 --annealing-epochs=50 --lr=1e-3 \
    --milestones=100 --resnet=18 --cifar=10 --verbose --progress-bar
```
The full list of command line options for the script is:
```
  --num-epochs: Total number of training epochs
  --train-samples: Number of MC samples to draw from the variational posterior during training for the data log likelihood  
  --test-samples: Number of samples to average the predictive posterior during testing.
  --annealing-epochs: Number of training epochs over which the weight of the KL term is annealed linearly.
  --ml-epochs: Number of training epochs where the weight of the KL term is 0.
  --inference-config: Path to the inference config file
  --output-dir: Directory in which to store state dicts of the network and optimizer, and the final calibration plot. 
  --verbose: Switch for printing validation accuracy and calibration at every epoch.
  --progress-bar: Switch for tqdm progress bar for epochs and batches.
  --lr: Initial learning rate.
  --seed: Random seed.
  --cifar: 10 or 100 for the corresponding CIFAR dataset.
  --optimizer: sgd or adam for the corresponding optimizer.
  --momentum: momentum if using sgd.
  --milestones: Comma-separated list of epochs after which to decay the learning rate by a factor of gamma. 
  --gamma: Multiplicative decay factor for the learning rate.
  --resnet: Which ResNet architecture from torchvision to use (must be one of {18, 34, 50, 101, 152}).
```


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
