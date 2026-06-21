# Datasets Guide — No Upload Required!

## All datasets auto-download. Total size: under 500 MB. No upload needed.

---

## Quick Reference: All Datasets Used (Total < 500 MB)

| Module | Dataset | Size | Download Method |
|--------|---------|------|-----------------|
| 01-02 | scikit-image built-ins | 0 MB (bundled) | `from skimage import data` |
| 01-06 | MNIST | 11 MB | `torchvision.datasets.MNIST(download=True)` |
| 02-12 | CIFAR-10 | 170 MB | `torchvision.datasets.CIFAR10(download=True)` |
| 05 | FashionMNIST | 30 MB | `torchvision.datasets.FashionMNIST(download=True)` |
| 05 | Pre-trained weights | ~50 MB | `models.resnet18(weights=...)` |
| 07 | BSD68 | 5 MB | URL download in notebook |
| 07 | Set5/Set14 | <1 MB | URL download in notebook |
| 08-09 | CIFAR-10 + synthetic masks | 0 MB extra | Generated from CIFAR-10 |
| 10 | MNIST + CIFAR-10 | 0 MB extra | Already downloaded above |
| 11 | Omniglot | 13 MB | `torchvision.datasets.Omniglot(download=True)` |
| 12 | MedMNIST | 50 MB | `pip install medmnist` then `medmnist.ChestMNIST()` |
| 12 | EuroSAT | 90 MB | `torchvision.datasets.EuroSAT(download=True)` |
| All | Synthetic data | 0 MB | Generated with numpy in each notebook |

**Maximum total download: ~420 MB (well under 500 MB limit)**

---

## How It Works

### In Google Colab:
```python
# Just run the notebook! Everything downloads automatically.
# Example from any notebook:

!pip install torch torchvision scikit-image -q

import torchvision
dataset = torchvision.datasets.CIFAR10(root='./data', download=True)
# ✅ Downloads automatically on first run!
```

### Locally:
```python
# Same code works locally. Data saves to ./data/ folder.
# First run downloads, subsequent runs use cached data.
```

---

## Recommended Approach for Learning

### Start Small (Modules 1-4):
- Use **scikit-image built-ins** (zero download, 5-10 images)
- Use **synthetic data** (numpy generated, infinite supply)
- Use **MNIST** (11 MB, tiny 28×28 images)

### Scale Up (Modules 5-8):
- Use **CIFAR-10** (170 MB, 32×32 color images)
- Use **FashionMNIST** (30 MB, clothing items)
- Use **subsets** (first 1000 images) for fast experiments

### Real Applications (Modules 9-12):
- Use **CIFAR-10 + synthetic annotations** for segmentation/detection
- Use **MedMNIST** (50 MB, real medical images)
- Use **EuroSAT** (90 MB, real satellite images)
- Use **Omniglot** (13 MB, meta-learning)

---

## Creating Synthetic Data (Zero Download!)

Most notebooks generate their own training data synthetically:

```python
import numpy as np

# Synthetic image enhancement dataset
def generate_enhancement_pair(size=64):
    """Generate (degraded, clean) image pair."""
    # Clean image: random patterns
    clean = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    
    # Add structured content (circles, gradients)
    for _ in range(3):
        cx, cy = np.random.randint(10, size-10, 2)
        r = np.random.randint(5, 15)
        color = np.random.randint(100, 255, 3)
        for i in range(size):
            for j in range(size):
                if (i-cy)**2 + (j-cx)**2 < r**2:
                    clean[i, j] = color
    
    # Degrade: add noise + darken
    degraded = clean.astype(np.float32) * 0.5  # darken
    degraded += np.random.normal(0, 20, degraded.shape)  # noise
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    
    return degraded, clean

# Generate 1000 training pairs instantly!
train_data = [generate_enhancement_pair() for _ in range(1000)]
```

---

## Google Colab Free Tier Limits

| Resource | Limit | Recommendation |
|----------|-------|----------------|
| RAM | 12 GB | Use subsets (first 1000-5000 images) |
| GPU RAM | 15 GB (T4) | Batch size 32-64 for training |
| Disk | 100 GB | Plenty for all datasets |
| Runtime | 12 hours | Save checkpoints frequently |

**All notebooks in this course are designed to run within Colab free tier limits!**
