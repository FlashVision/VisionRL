![Module Logo](../logo.png)

# 12.4 — Video Frame Enhancement

## Overview

Video enhancement extends image enhancement with a critical new dimension: **temporal consistency**. Enhancing each frame independently produces flickering artifacts — the agent must learn to balance per-frame quality with smooth transitions between frames. This project builds an RL agent that enhances video frames while maintaining temporal coherence, using optical flow for motion estimation, a recurrent state for memory, and a temporal consistency reward term. The "video" is synthesized from CIFAR-10/MNIST frames with programmatic motion.

## Mathematical Foundation

### 1. Temporal Consistency Loss

Given consecutive enhanced frames $\hat{I}_t$ and $\hat{I}_{t-1}$, and the optical flow $\mathbf{u}_{t \to t-1}$ from frame $t$ to $t-1$, the temporal consistency loss is:

$$\mathcal{L}_{\text{tc}} = \left\|\hat{I}_t - \text{warp}(\hat{I}_{t-1}, \mathbf{u}_{t \to t-1})\right\|^2$$

where the **warping** operation moves pixels according to the flow field:

$$\text{warp}(I, \mathbf{u})(x, y) = I(x + u_x(x,y), \; y + u_y(x,y))$$

implemented via bilinear interpolation for sub-pixel accuracy.

**Intuition**: If a pixel in frame $t$ corresponds to a pixel at a different position in frame $t-1$ (due to motion), the enhanced values at these corresponding locations should be consistent.

### 2. Optical Flow: Horn-Schunck Method

Optical flow estimates per-pixel motion between frames. The **brightness constancy assumption**:

$$I(x, y, t) = I(x + u_x, y + u_y, t + 1)$$

Taylor expansion:

$$I(x, y, t) \approx I(x, y, t) + I_x u_x + I_y u_y + I_t$$

giving the **optical flow constraint equation**:

$$I_x u_x + I_y u_y + I_t = 0 \quad \Leftrightarrow \quad \nabla I \cdot \mathbf{u} + I_t = 0$$

This is one equation with two unknowns ($u_x, u_y$) — the **aperture problem**. Horn-Schunck adds a **smoothness regularizer**:

$$E(\mathbf{u}) = \iint \left[(I_x u_x + I_y u_y + I_t)^2 + \lambda\left(\|\nabla u_x\|^2 + \|\nabla u_y\|^2\right)\right] dx \, dy$$

**Euler-Lagrange equations**:

$$I_x(I_x u_x + I_y u_y + I_t) - \lambda \nabla^2 u_x = 0$$

$$I_y(I_x u_x + I_y u_y + I_t) - \lambda \nabla^2 u_y = 0$$

**Iterative solution** (Jacobi/Gauss-Seidel):

$$u_x^{(n+1)} = \bar{u}_x^{(n)} - \frac{I_x(I_x \bar{u}_x^{(n)} + I_y \bar{u}_y^{(n)} + I_t)}{\lambda + I_x^2 + I_y^2}$$

$$u_y^{(n+1)} = \bar{u}_y^{(n)} - \frac{I_y(I_x \bar{u}_x^{(n)} + I_y \bar{u}_y^{(n)} + I_t)}{\lambda + I_x^2 + I_y^2}$$

where $\bar{u}$ denotes the local average (Laplacian approximation).

### 3. Recurrent State for Video RL Agent

The agent maintains a **recurrent hidden state** $h_t$ that captures temporal context:

$$h_t = \text{GRU}(h_{t-1}, \; \phi(I_t))$$

where $\phi(I_t)$ is a CNN feature encoding of the current frame and GRU is a Gated Recurrent Unit:

$$z_t = \sigma(W_z [\phi(I_t), h_{t-1}]) \quad \text{(update gate)}$$

$$r_t = \sigma(W_r [\phi(I_t), h_{t-1}]) \quad \text{(reset gate)}$$

$$\tilde{h}_t = \tanh(W_h [\phi(I_t), r_t \odot h_{t-1}]) \quad \text{(candidate)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

The policy then conditions on the recurrent state:

$$a_t = \pi_\theta(I_t, h_t)$$

This allows the agent to remember past enhancement decisions and ensure consistency.

### 4. Frame-to-Frame Reward Smoothing

The total per-frame reward combines quality and consistency:

$$r_t = \underbrace{\text{PSNR}(\hat{I}_t, I_t^{\text{gt}})}_{\text{quality}} - \underbrace{\mu \cdot \mathcal{L}_{\text{tc}}(t)}_{\text{temporal penalty}} + \underbrace{\nu \cdot \text{SSIM}(\hat{I}_t, I_t^{\text{gt}})}_{\text{structural}}$$

**Exponential smoothing** of rewards to prevent drastic changes:

$$\bar{r}_t = (1 - \alpha_{\text{smooth}}) \cdot \bar{r}_{t-1} + \alpha_{\text{smooth}} \cdot r_t$$

**Temporal difference reward** — penalize large reward changes between frames:

$$r_t^{\text{final}} = r_t - \eta \cdot |r_t - r_{t-1}|$$

### 5. Video Enhancement MDP

| Component | Definition |
|-----------|-----------|
| **State** $s_t$ | $(I_t, h_t, \hat{I}_{t-1}, \mathbf{u}_{t \to t-1})$ — current frame, recurrent state, previous enhanced frame, flow |
| **Action** $a_t$ | Enhancement parameters: $(\Delta\text{brightness}, \Delta\text{contrast}, \sigma_{\text{denoise}}, \text{sharpness})$ |
| **Reward** $r_t$ | PSNR improvement - $\mu \cdot \mathcal{L}_{\text{tc}}$ |
| **Episode** | One video sequence ($T$ frames) |
| **Key constraint** | Enhancement parameters should change smoothly: $\|a_t - a_{t-1}\| < \delta$ |

## Step-by-Step Breakdown

1. **Generate synthetic video**: Create sequences by translating CIFAR-10/MNIST images across frames (horizontal motion, rotation, zoom).
2. **Add temporal degradation**: Apply noise + blur that varies smoothly over time.
3. **Implement Horn-Schunck optical flow**: Compute frame-to-frame motion fields.
4. **Build recurrent RL agent**: CNN encoder + GRU + MLP policy head.
5. **Define temporal consistency reward**: PSNR improvement minus temporal flickering penalty.
6. **Train on synthetic videos**: Agent learns to enhance frames while maintaining temporal coherence.
7. **Evaluate**: Compare per-frame metrics (PSNR, SSIM) and temporal consistency (warping error) against frame-independent baseline.

## Dataset Used

| Property | Value |
|----------|-------|
| **Name** | Synthetic video sequences |
| **Source** | Generated from CIFAR-10 and MNIST (auto-download base datasets) |
| **Auto-download** | `CIFAR10(root='./data', download=True)` for base frames |
| **Generation** | Apply affine transforms (translation, rotation) to create frame sequences |
| **Sequence length** | 10–30 frames per video |
| **Why** | Full control over motion patterns; ground truth available; no large video dataset needed |

```python
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
import torch.nn.functional as F

base_dataset = CIFAR10(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)

def generate_video_sequence(base_image, num_frames=15, motion='translate'):
    """Generate a synthetic video by applying motion to a base image."""
    C, H, W = base_image.shape
    frames = []
    pad_img = F.pad(base_image, (W//2, W//2, H//2, H//2), mode='reflect')
    
    for t in range(num_frames):
        if motion == 'translate':
            dx = int(t * 1.5)
            dy = int(t * 0.5)
        elif motion == 'zoom':
            scale = 1.0 + t * 0.02
            dx, dy = 0, 0
        
        cx, cy = W//2 + W//2 + dx, H//2 + H//2 + dy
        frame = pad_img[:, cy-H//2:cy+H//2, cx-W//2:cx+W//2]
        
        sigma_t = 0.05 + 0.01 * t  # increasing noise
        frame = frame + torch.randn_like(frame) * sigma_t
        frames.append(frame.clamp(0, 1))
    
    return torch.stack(frames)  # (T, C, H, W)

# Generate a sample video
base_img, _ = base_dataset[0]
video = generate_video_sequence(base_img, num_frames=15)
print(f"Video shape: {video.shape}")  # (15, 3, 32, 32)
```

## Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\mathcal{L}_{\text{tc}} = \|\hat{I}_t - \text{warp}(\hat{I}_{t-1}, \mathbf{u})\|^2$ | Temporal consistency loss |
| $\nabla I \cdot \mathbf{u} + I_t = 0$ | Optical flow constraint |
| $E = \iint [(I_x u_x + I_y u_y + I_t)^2 + \lambda(\|\nabla u_x\|^2 + \|\nabla u_y\|^2)] \, dA$ | Horn-Schunck energy |
| $h_t = \text{GRU}(h_{t-1}, \phi(I_t))$ | Recurrent agent state |
| $r_t = \text{PSNR}_t - \mu\mathcal{L}_{\text{tc}} + \nu\text{SSIM}_t$ | Video enhancement reward |

## Connection to RL

Video enhancement is a compelling RL problem because:

- **Temporal structure**: Each frame decision affects the next — a fundamentally sequential problem that benefits from recurrent policies.
- **Conflicting objectives**: Per-frame quality vs. temporal consistency creates a trade-off that the RL agent learns to balance.
- **Memory**: The recurrent state allows the agent to "remember" past enhancements and maintain consistency — standard image enhancement methods lack this.
- **Non-stationarity**: Video content changes over time (scene changes, lighting shifts) — the agent must adapt its enhancement strategy.
- **Real-time constraint**: In deployment, the agent must process frames quickly and consistently — RL can optimize for this via reward shaping.

## Prerequisites & Next Steps

**Prerequisites:**
- Module 7: Image Enhancement (per-frame operations)
- Module 6: PPO with recurrent policies
- Basic computer vision (optical flow concepts)

**Next Steps:**
- → 12.5 Complete Vision Pipeline: Integrate video enhancement with detection and segmentation
- → Deployment: Adapt for real video streams (webcam, surveillance)
