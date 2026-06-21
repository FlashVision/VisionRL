![Module Logo](../logo.png)

# Datasets for Module 01: Image Foundations

## No Upload Required! All datasets auto-download.

### Datasets Used:

| Dataset | Source | Size | Auto-Download |
|---------|--------|------|---------------|
| scikit-image built-ins | `skimage.data` | ~50 images | `from skimage import data; img = data.astronaut()` |
| MNIST Digits | torchvision | 70K images (28×28) | `torchvision.datasets.MNIST(root='./data', download=True)` |
| Synthetic | Generated in code | Variable | Created with numpy |

### Quick Usage:

```python
# scikit-image (no download needed, comes with the package)
from skimage import data
astronaut = data.astronaut()      # 512×512×3 RGB
camera = data.camera()            # 512×512 grayscale
coins = data.coins()              # 303×384 grayscale
coffee = data.coffee()            # 400×600×3 RGB

# MNIST (auto-downloads ~11MB on first use)
import torchvision
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)

# Synthetic (generated on-the-fly, zero download)
import numpy as np
synthetic_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
```

### Why These Datasets?

- **scikit-image built-ins**: Perfect for learning basics. Small, diverse, well-known images.
- **MNIST**: Tiny images (28×28) ideal for understanding pixel structure and matrix operations.
- **Synthetic**: Full control over image properties for demonstrating specific concepts.
