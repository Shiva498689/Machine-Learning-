# 🎧 The Frequency Quest  
### *Conditional WGAN-GP Audio Generation with HiFi-GAN Vocoder*  

![Python](https://img.shields.io/badge/Python-3.10-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange)  
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Overview

**The Frequency Quest** is a deep learning project that combines **Conditional Generative Adversarial Networks (CGAN)** with **Wasserstein loss and Gradient Penalty (WGAN-GP)** to generate **Mel-spectrograms** conditioned on class labels.  
The generated Mel-spectrograms are then converted into audible sound using the **HiFi-GAN Vocoder** for realistic, high-quality audio synthesis.  

This project is ideal for learning, research, and experimentation in **neural audio generation**, **GAN stability**, and **conditional generative modeling**.

---

## ⚙️ Key Features

- 🎛️ **WGAN-GP Framework** — Stable GAN training with gradient penalty for better convergence  
- 🎚️ **Conditional Generation** — Class-based audio generation using label embeddings  
- 🧩 **HiFi-GAN Integration** — Converts Mel-spectrograms into high-fidelity `.wav` audio  
- 🗃️ **Custom Dataset Loader** — Converts `.wav` to Mel-spectrograms and pads/trims automatically  
- 📊 **Training Visualization** — Saves loss curves, generated samples, and spectrograms after each epoch  

---

## 🧩 Architecture Overview

### 🎨 **1️⃣ Generator**
- Input → latent noise vector `z` + one-hot encoded class label  
- Output → 80×512 Mel-spectrogram  
- Layers → Dense + ConvTranspose2D + ReLU  
- Output activation → ReLU (to match `log1p` Mel scale)

### 🧱 **2️⃣ Discriminator**
- Input → Mel-spectrogram + label embedding  
- Output → Wasserstein critic score (realness measure)  
- Layers → Conv2D + LeakyReLU  
- No BatchNorm (as per WGAN-GP design)

### 🔊 **3️⃣ HiFi-GAN Vocoder**
- Converts generated Mel-spectrograms into `.wav` audio  
- Loaded via `torchaudio.prototype.pipelines` pretrained HiFi-GAN  

---

## 📦 Dataset Preparation

Your dataset must be structured like this:

```
dataset/
 ├── train/
 │    ├── class_1/
 │    │     ├── file1.wav
 │    │     ├── file2.wav
 │    ├── class_2/
 │    │     ├── file1.wav
 │    │     ├── file2.wav
 │    ...
```

Each subfolder represents a **class label**.  
The model automatically encodes them as integer labels during training.

---

## 🧮 Training Details

| Parameter | Value |
|------------|--------|
| Model Type | Conditional WGAN-GP |
| Latent Dim | 100 |
| n_critic | 5 |
| λ_gp | 10 |
| Learning Rate | 2e-4 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Batch Size | 128 |
| n_mels | 80 |
| Frames per Sample | 512 |
| Sample Rate | 22050 Hz |

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
!pip install torch torchaudio torchvision tqdm matplotlib
```

### 2️⃣ Set Dataset Path

Edit this line in the code:
```python
BASE_PATH = '/kaggle/input/the-frequency-quest/the-frequency-quest - Copy/train'
```

### 3️⃣ Train the Model

```bash
python train_audio_gan.py
```

### 4️⃣ During Training

Outputs are saved automatically:
```
gan_generated_audio/     → Generated audio clips (.wav)
gan_spectrogram_plots/   → Spectrograms each epoch
gan_loss_plot.png        → Generator & Discriminator loss curves
```

---

## 🎨 Visual & Audio Outputs

| Output Type | Description |
|--------------|-------------|
| 🖼️ Spectrogram | Visualizes generator output over training |
| 🎧 Generated Audio | Converted `.wav` clips using HiFi-GAN |
| 📉 Loss Curves | Shows training stability over epochs |

---

## 📈 Example Output Timeline

| Epoch | Quality | Description |
|--------|----------|-------------|
| 1 | 🟠 Rough noise | Initial random audio |
| 50 | 🟡 Semi-structured | Basic tonal patterns |
| 100+ | 🟢 Realistic | Clear class-conditioned audio |

---

## 🧠 Core Concepts Used

- **Conditional GANs (CGAN)** — Learn label-conditioned generation  
- **WGAN-GP** — Uses Wasserstein distance + gradient penalty for stability  
- **HiFi-GAN Vocoder** — Converts Mel-spectrograms into realistic waveforms  
- **Mel-Spectrogram Representation** — Frequency vs time with perceptual scaling  

---

## 📂 Repository Structure

```
├── train_audio_gan.py          # Main training script
├── gan_generated_audio/        # Generated samples
├── gan_spectrogram_plots/      # Spectrogram visualizations
├── gan_loss_plot.png           # Loss graph
├── README.md                   # Project documentation
```

---

## 🧑‍💻 Authors & Contributors

| Name | Role | Institution |
|------|------|--------------|
| **Shiva Dubey** | Lead Developer | IIT Indore |
 — |

---

## 🧾 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute this for educational and research purposes.

---

## ⭐ Acknowledgements

- [HiFi-GAN: High-Fidelity Neural Vocoder](https://github.com/jik876/hifi-gan)  
- [Torchaudio](https://pytorch.org/audio/stable/index.html)  
- [WGAN-GP (Gulrajani et al., 2017)](https://arxiv.org/abs/1704.00028)  
- [Conditional GANs (Mirza & Osindero, 2014)](https://arxiv.org/abs/1411.1784)

---

<p align="center">
  <b>🎧 The Frequency Quest — Redefining Audio Generation</b><br>
  <sub>Built with ❤️ using PyTorch and HiFi-GAN</sub>
</p>
