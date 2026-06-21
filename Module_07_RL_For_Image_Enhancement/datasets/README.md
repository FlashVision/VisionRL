![Module Logo](../logo.png)

# Datasets for Module 07: RL for Image Enhancement

## No Upload Required! All datasets auto-download.

### Datasets Used:

| Dataset | Source | Size | Use Case | Auto-Download |
|---------|--------|------|----------|---------------|
| CIFAR-10 | torchvision | 170 MB | Base images for degradation | `torchvision.datasets.CIFAR10(download=True)` |
| BSD68 | URL download | 5 MB | Denoising benchmark (68 images) | `wget` in notebook |
| Set5 / Set14 | URL download | <1 MB | Super-resolution benchmark | `wget` in notebook |
| Synthetic HDR | Generated | Variable | HDR tone mapping | numpy code |

### How Degradation Works (No Upload Needed):

```python
import torchvision
import numpy as np

# 1. Download clean images
cifar10 = torchvision.datasets.CIFAR10(root='./data', download=True)

# 2. Programmatically degrade them (this is our "input")
def degrade_brightness(img, factor=0.4):
    """Simulate underexposed image"""
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def add_noise(img, sigma=25):
    """Add Gaussian noise"""
    return np.clip(img + np.random.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def add_color_cast(img, gains=(1.3, 0.8, 0.9)):
    """Simulate wrong white balance"""
    result = img.copy().astype(np.float32)
    for c, g in enumerate(gains):
        result[:,:,c] *= g
    return np.clip(result, 0, 255).astype(np.uint8)

# 3. RL agent learns to restore: degraded → clean
# Ground truth = original CIFAR-10 image
# Input = degraded version
# Reward = PSNR(agent_output, ground_truth)
```

### BSD68 Auto-Download:
```python
import urllib.request
import zipfile

url = "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"
urllib.request.urlretrieve(url, "bsd68.zip")
with zipfile.ZipFile("bsd68.zip", 'r') as zip_ref:
    zip_ref.extractall("./data/bsd68")
```

### Why These Datasets?

- **CIFAR-10 + degradation**: Infinite training data by applying random degradations
- **BSD68**: Standard denoising benchmark used in all research papers
- **Set5/Set14**: Classic super-resolution benchmarks (tiny, fast evaluation)
- **Synthetic HDR**: Full control over dynamic range for tone mapping experiments
