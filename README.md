# pytorch-named-dims

PyTorch tensor dimension names for all `nn.Modules`.

Extends [PyTorch Named Tensors](https://pytorch.org/docs/stable/named_tensor.html) ([new in PyTorch 1.4.0](https://github.com/pytorch/pytorch/releases/tag/v1.4.0), still experimental as of PyTorch 1.5.0). It works in Python 3.6+.

Inspired by:

* [Quantum Tensors JS](https://github.com/Quantum-Game/quantum-tensors) by Piotr Migdał
* [Tensor Considered Harmful](http://nlp.seas.harvard.edu/NamedTensor) by Alexander Rush

## Installation

Not yet on PyPI. Install:

```bash
pip install git+git://github.com/stared/pytorch-named-dims.git
```

## Example

```python
import torch
from torch import nn
from pytorch_named_dims import nm

convs = nn.Sequential(
    nm.Conv2d(3, 5, kernel_size=3, padding=1),
    nn.ReLU(),  # preserves dims on its own
    nm.MaxPool2d(2, 2),
    nm.Conv2d(5, 2, kernel_size=3, padding=1)
)

x_input_1 = torch.rand((4, 3, 2, 2), names=('N', 'C', 'H', 'W'))  # good
x_input_2 = torch.rand((4, 3, 2, 2), names=('N', 'C', 'W', 'H'))  # bad

convs(x_input_1)  # returns ('N', 'C', 'H', 'W')
convs(x_input_2)  # raises:
# Layer Conv2d requires dimensions ['N', 'C', 'H', 'W'] but got ('N', 'C', 'W', 'H') instead.
```

* TODO: Colab

## Funding

Project is supported by [Program Operacyjny Inteligentny Rozwój grant for ECC Games for GearShift project](https://mapadotacji.gov.pl/projekty/874596/?lang=en).
