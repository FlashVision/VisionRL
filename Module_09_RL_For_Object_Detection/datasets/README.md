![Module Logo](../logo.png)

# Datasets for Module 09: RL for Object Detection

## No Upload Required! All datasets auto-download.

### Datasets Used:

| Dataset | Source | Size | Use Case | Auto-Download |
|---------|--------|------|----------|---------------|
| MNIST on Canvas | Generated from MNIST | 11 MB | Object localization | `torchvision.datasets.MNIST(download=True)` |
| CIFAR-10 | torchvision | 170 MB | Real photos | `torchvision.datasets.CIFAR10(download=True)` |
| Multi-digit Scenes | Generated | 0 B | Multi-object detection | numpy + MNIST |
| Simple VQA | Generated | 0 B | Visual QA | Synthetic shapes |

### Quick Usage:

```python
import torchvision
import numpy as np
from PIL import Image

# MNIST digits placed on larger canvas (for localization)
mnist = torchvision.datasets.MNIST(root='./data', download=True)

def create_detection_scene(canvas_size=128, n_objects=3):
    """Place MNIST digits randomly on canvas with bounding boxes."""
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    boxes = []
    labels = []
    
    for _ in range(n_objects):
        idx = np.random.randint(len(mnist))
        digit_img, label = mnist[idx]
        digit = np.array(digit_img)
        
        # Random position
        x = np.random.randint(0, canvas_size - 28)
        y = np.random.randint(0, canvas_size - 28)
        
        canvas[y:y+28, x:x+28] = np.maximum(canvas[y:y+28, x:x+28], digit)
        boxes.append([x, y, x+28, y+28])  # [x1, y1, x2, y2]
        labels.append(label)
    
    return canvas, boxes, labels

# Simple VQA dataset (synthetic)
def create_vqa_sample():
    """Create image with colored shape + question."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    colors = {'red': [255,0,0], 'green': [0,255,0], 'blue': [0,0,255]}
    shapes = ['circle', 'square', 'triangle']
    
    color_name = np.random.choice(list(colors.keys()))
    shape = np.random.choice(shapes)
    
    # Draw shape with color
    if shape == 'circle':
        cv2.circle(img, (32, 32), 15, colors[color_name], -1)
    elif shape == 'square':
        img[17:47, 17:47] = colors[color_name]
    
    question = f"What color is the {shape}?"
    answer = color_name
    
    return img, question, answer
```

### Why These Datasets?

- **MNIST on Canvas**: Known ground truth boxes, controllable difficulty
- **Multi-digit scenes**: Multi-object detection with varying object count
- **Synthetic VQA**: Simple enough to learn, demonstrates attention mechanism
