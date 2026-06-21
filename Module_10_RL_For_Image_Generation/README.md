![Module Logo](logo.png)

# Module 10: RL for Image Generation

## Overview

This module explores how Reinforcement Learning can drive **creative image generation tasks** — from training GANs and performing neural style transfer to inpainting missing regions, aligning text-to-image models with human preferences, and teaching agents to paint from scratch. Each sub-module formulates generation as a sequential decision process where an RL agent learns to produce or refine visual content by maximizing a quality-based reward signal.

## Sub-Modules

| # | Topic | Key Idea |
|---|-------|----------|
| 10.1 | [RL with GANs](01_RL_With_GANs/) | RL agent controls GAN training dynamics |
| 10.2 | [Neural Style Transfer RL](02_Neural_Style_Transfer_RL/) | Agent balances content vs. style loss |
| 10.3 | [Image Inpainting Agent](03_Image_Inpainting_Agent/) | Agent selects fill strategies per region |
| 10.4 | [Text-to-Image RL (RLHF)](04_Text_To_Image_RL/) | Align generation with human preferences |
| 10.5 | [Creative Image Agent](05_Creative_Image_Agent/) | Agent learns to paint digit strokes |

## Mathematical Theme

All sub-modules share a common formulation: the **generation process is an MDP** where:

$$s_t = \text{current generated image (or latent)}, \quad a_t = \text{generation parameter adjustment}$$

$$r_t = \mathcal{Q}(s_t, s_{\text{target}}) \quad \text{(quality metric: FID, LPIPS, CLIP score, L2, etc.)}$$

$$\pi^* = \arg\max_\pi \; \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

## Prerequisites

- Modules 1–6 (image fundamentals + Deep RL)
- Modules 7–9 (RL for enhancement, segmentation, detection)
- Basic understanding of generative models (GANs, diffusion)

## Learning Outcomes

After completing this module you will be able to:

1. Formulate image generation tasks as RL problems
2. Use RL to guide GAN training and avoid mode collapse
3. Apply RLHF principles to align visual generation with human preferences
4. Build agents that paint, inpaint, and stylize images autonomously
