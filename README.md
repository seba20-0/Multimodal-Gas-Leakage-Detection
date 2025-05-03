# Gas Detection with Multimodal Bayesian and MC-Dropout Uncertainty

A comprehensive framework for detecting gas types by fusing thermal camera images and gas sensor readings, using classical neural networks, Bayesian neural networks, and Monte Carlo Dropout for uncertainty estimation. This repository provides:

* **Sensor-only**, **Image-only**, and **Multimodal** models
* **Bayesian neural network** variants for sensors, images, and multimodal network with TensorFlow Probability
* **RandomSensorDropout** for robust simulation of sensor failure
* **MC-Dropout inference** for uncertainty quantification
* Optimized **`tf.data`** pipelines for CSV and image loading

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gas-detection.git
cd gas-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train a model

#### Sensor-only example:

```bash
python src/train.py sensor \
  --sensor-data data/sensor.csv \
  --output checkpoints/sensor_model \
  --epochs 30 \
  --sensor-dropout 0.3 \
  --layer-dropout 0.5 \
  --hidden-units 64 32
```

#### Image-only example:

```bash
python src/train.py image \
  --image-data-dir data/images \
  --output checkpoints/image_model \
  --epochs 30 \
  --layer-dropout 0.5 \
  --conv-filters 32 64 \
  --dense-units 128
```

#### Multimodal Fusion example:

```bash
python src/train.py fusion \
  --sensor-data data/sensor.csv \
  --image-data-dir data/images \
  --output checkpoints/fusion_model \
  --epochs 30 \
  --sensor-dropout 0.3 \
  --layer-dropout 0.5 \
  --sensor-units 32 \
  --img-dense 64 \
  --fusion-dense 128
```

### 4. Evaluate and visualize

```bash
python src/evaluate.py \
  --model best_model.keras \
  --mc-samples 50 \
  --output figures/uncertainty_plot.png
```

---

## 📂 Repository Structure

```plaintext
gas-detection/
├── dataset/
│   ├── Gas Sensors Measurement/      # CSV for gas readings
│   └── Thermal Camera Images/        # Images of 4 classes
│       ├── Mixture
│       ├── NoGas
│       ├── Perfume
│       └── Smoke
│
├── src/                             # source code
│   ├── data_loader.py               # CSV & image tf.data pipelines
│   ├── models.py                    # NN & Bayesian model builders
│   ├── train.py                     # training scripts with CLI args
│   ├── evaluate.py                  # normal + MC-Dropout evaluation
│
├── notebooks/                       # exploration & plots
│   └── ablation_study.ipynb
│
├── requirements.txt                 # pinned Python dependencies
├── README.md                        # project overview & instructions
└── LICENSE                          # Apache License
```

---

## 📑 Data Sources

We use the publicly available multimodal gas dataset of thermal images and sensor readings:

* **Mendeley Data**: "Multimodal Gas Sensing and Thermal Imaging Dataset" (Version 2), Mendeley Data, 2024. doi:10.17632/zkwgkjkjn9.2

Please cite the original dataset in any publications:

> T. Samal and A. Ghosh, *Multimodal Gas Sensing and Thermal Imaging Dataset* (Version 2), Mendeley Data, 2024.

## Contributions

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes
4. Open a Pull Request

---

## License

This project is licensed under the [Apache License](LICENSE).
