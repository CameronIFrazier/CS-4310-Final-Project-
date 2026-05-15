# CS-4310-Final-Project-
# DCGAN on MNIST
### Machine Learning and Its Applications — Final Project

A PyTorch implementation of a **Deep Convolutional GAN (DCGAN)** trained on the MNIST handwritten digits dataset, based on [Radford et al. (2015)](https://arxiv.org/abs/1511.06434).

---

## How to Run

1. **Clone the repository** and navigate to the project folder:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies** (see below).

3. **Open and run the notebook:**
   ```bash
   jupyter notebook DCGAN.ipynb
   ```
   Run all cells top to bottom. The MNIST dataset will be downloaded automatically on first run into a `./data/` directory.

> **GPU recommended.** Training runs for 50 epochs. The notebook auto-detects CUDA and falls back to CPU if unavailable.

---

## Required Packages / Dependencies

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `torchvision` | MNIST dataset + image utilities |
| `matplotlib` | Plotting loss curves and image grids |
| `numpy` | Array operations |

Install all dependencies with:
```bash
pip install torch torchvision matplotlib numpy
```

Or with conda:
```bash
conda install pytorch torchvision matplotlib numpy -c pytorch
```


## Main Code Locations

All training, inference, and demo code lives in **`DCGAN.ipynb`**:

| Section | Description |
|---|---|
| **Section 1 — Setup** | Imports, device configuration, output directory setup |
| **Section 2 — Hyperparameters** | All tunable settings (latent dim, batch size, epochs, LR, etc.) |
| **Section 3 — Dataset** | MNIST loading, preprocessing, real image visualization |
| **Section 4 — Model Architecture** | Generator and Discriminator class definitions |
| **Section 5 — Loss & Optimizers** | BCE loss, Adam optimizers, fixed noise vector |
| **Section 6 — Training Loop** | ⬅️ **Main training code** |
| **Section 7 — Results** | Loss curves, generated image progression, real vs. fake comparison |
| **Section 8 — Latent Space Interpolation** | ⬅️ **Demo / inference code** |
| **Section 9 — Save Model** | Exports `generator_mnist.pth` and `discriminator_mnist.pth` |
| **Section 10 — Summary** | Discussion of results and design decisions |

---

## References

- Goodfellow et al., *Generative Adversarial Nets* (NeurIPS 2014): https://arxiv.org/abs/1406.2661
- Radford et al., *Unsupervised Representation Learning with Deep Convolutional GANs* (2015): https://arxiv.org/abs/1511.06434
