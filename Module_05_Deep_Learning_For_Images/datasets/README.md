![Module Logo](../logo.png)

# Datasets for Module 05: Deep Learning for Images

## No Upload Required! All datasets auto-download.

### Datasets Used:

| Dataset | Source | Size | Images | Auto-Download |
|---------|--------|------|--------|---------------|
| MNIST | torchvision | 11 MB | 70K (28×28) | `torchvision.datasets.MNIST(download=True)` |
| CIFAR-10 | torchvision | 170 MB | 60K (32×32×3) | `torchvision.datasets.CIFAR10(download=True)` |
| FashionMNIST | torchvision | 30 MB | 70K (28×28) | `torchvision.datasets.FashionMNIST(download=True)` |
| ImageNet pre-trained | torchvision.models | ~100 MB | N/A (weights only) | `models.resnet18(pretrained=True)` |

### Quick Usage:

```python
import torchvision
import torchvision.transforms as transforms

# Transform for normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 (main dataset for this module)
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# For quick experiments, use a small subset:
small_dataset = torch.utils.data.Subset(trainset, range(1000))

# Pre-trained models (weights auto-download)
from torchvision import models
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
```

### Why These Datasets?

- **MNIST**: Simplest image classification benchmark (10 digit classes)
- **CIFAR-10**: Standard benchmark, 10 classes of natural images, small enough for Colab
- **FashionMNIST**: Clothing items, great for transfer learning (MNIST→Fashion domain shift)
- **Pre-trained weights**: No training on ImageNet needed, just download the weights
