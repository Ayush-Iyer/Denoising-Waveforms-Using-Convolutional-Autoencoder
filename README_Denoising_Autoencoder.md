# 🧠 1D Convolutional Denoising Autoencoder with TensorFlow & Keras

This repository contains the implementation of a **Denoising Autoencoder (DAE)** for 1D ECG-like signal data using Keras and TensorFlow. The model is designed to reconstruct clean signals from various noisy inputs using convolutional and transposed convolutional neural networks.

---

## 📌 Overview

This Denoising Autoencoder project:

- Uses **random Gaussian, Salt & Pepper, Poisson, and Uniform noise** during training
- Handles **1D time-series data (ECG signals)** from CSV input
- Applies **custom noise generation functions**
- Implements **transpose convolution via Conv2DTranspose** for decoding
- Uses **skip connections and data normalization** for enhanced learning
- Evaluates performance using **MAE, RMSE, and SSIM**

---

## ⚙️ Requirements

To run this project, install the following dependencies:

```bash
pip install tensorflow>=2.11.0
pip install keras>=2.11.0
pip install numpy pandas matplotlib seaborn
pip install scikit-image opencv-python
```

---

## 📂 Dataset

The dataset used in this project is a CSV file (`ecg.csv`) containing ECG-like time-series signals. Each row represents one sample, with 140 signal values and a label (unused in training).

Originally, this dataset may represent classification targets, but here it is **repurposed for denoising and signal reconstruction**.

---

## ✨ Features

- 4 types of synthetic noise injection:
  - Gaussian
  - Salt & Pepper
  - Poisson
  - Uniform
- Convolutional Encoder and Transpose Convolution Decoder
- Custom hybrid loss: MSE + MAE
- Real-time denoising evaluation on 1D signals
- Visual loss plots per noise variant

---

## 🧠 Model Architecture

- **Encoder**: Stack of Conv1D + BatchNorm + MaxPooling layers
- **Decoder**: Conv2DTranspose (simulated 1D transpose conv) with cropping
- **Skip connections** via Lambda layers
- **Final Layer**: Sigmoid activated output for pixel-level signal reconstruction

---

## 📏 Evaluation Metrics

- **MAE (Mean Absolute Error)**  
- **MSE (Mean Squared Error)**  
- **RMSE (Root Mean Squared Error)**  
- **SSIM (Structural Similarity Index)** – adapted for 1D signals

---

## 📈 Results

Each noise type is trained independently. Example metrics include:

| Noise Type     | MAE    | MSE    | RMSE   | SSIM   |
|----------------|--------|--------|--------|--------|
| Gaussian       | ~0.025 | ~0.001 | ~0.032 | ~0.91  |
| Salt & Pepper  | ~0.035 | ~0.002 | ~0.045 | ~0.87  |
| Poisson        | ~0.028 | ~0.001 | ~0.036 | ~0.89  |
| Uniform        | ~0.030 | ~0.001 | ~0.038 | ~0.88  |

> Note: Actual results may vary based on system and training seed.

---

## 📉 Visualization

Training loss is visualized using Seaborn for each noise variant:

- Smoothed validation loss curves
- Easy comparison of convergence behavior

---

## 🔮 Future Enhancements

- Add residual connections for deeper architectures
- Integrate real-world noise profiles (from sensors)
- Use GRU/LSTM layers for temporal dependencies
- Experiment with attention-based autoencoders

---

## 🤝 Contributing

We welcome contributions! Feel free to:

⭐ Star this repository if you find it helpful  
🐛 Open issues for suggestions or bugs  
🔀 Submit pull requests with improvements

---

## 📬 Contact

For questions or collaborations, feel free to reach out via [GitHub Issues](https://github.com/).

---

🚀 **Empowering Signal Denoising with Deep Learning!** 📡
