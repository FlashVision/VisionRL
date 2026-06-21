![Module Logo](../logo.png)

# Datasets for Module 08: RL for Image Segmentation

## No Upload Required! All datasets auto-download. Total < 200 MB.

### Datasets Used:

| Dataset | Source | Size | Use Case | Auto-Download |
|---------|--------|------|----------|---------------|
| CIFAR-10 | torchvision | 170 MB | Real photos + pseudo-masks | `torchvision.datasets.CIFAR10(download=True)` |
| FashionMNIST | torchvision | 30 MB | Clothing segmentation | `torchvision.datasets.FashionMNIST(download=True)` |
| Synthetic Shapes | Generated | 0 B | Simple segmentation | numpy code |

### Quick Usage:

```python
import torchvision
import numpy as np

# CIFAR-10 real photos (170 MB, likely already cached)
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# FashionMNIST (30 MB - tiny!)
fashion = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)

# Generate pseudo-segmentation masks from real images
def generate_pseudo_masks(dataset, n=500):
    masks = []
    for i in range(n):
        img = np.array(dataset[i][0])
        gray = np.mean(img, axis=2) if img.ndim == 3 else img.astype(float)
        mask = (gray > np.median(gray)).astype(np.uint8)
        masks.append(mask)
    return masks

# Synthetic shapes (RECOMMENDED for learning - zero download, instant)
def create_segmentation_dataset(n_samples=100, size=64):
    """Generate synthetic images with known segmentation masks."""
    images = []
    masks = []
    for _ in range(n_samples):
        img = np.random.randint(50, 150, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        
        # Random circle
        cx, cy = np.random.randint(15, size-15, 2)
        r = np.random.randint(8, 20)
        for i in range(size):
            for j in range(size):
                if (i-cy)**2 + (j-cx)**2 < r**2:
                    img[i, j] = [200, 50, 50]  # Red object
                    mask[i, j] = 1  # Object class
        
        images.append(img)
        masks.append(mask)
    return images, masks

# Use synthetic for fast experiments
images, masks = create_segmentation_dataset(100)
```

### Why These Datasets?

- **Synthetic shapes**: Perfect ground truth, instant generation, ideal for understanding RL segmentation
- **CIFAR-10 + pseudo-masks**: Real photos with auto-generated masks (no large download!)
- **FashionMNIST**: Clear foreground/background separation, only 30 MB
