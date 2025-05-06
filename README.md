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
cd Multimodal-Gas-Leakage-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train a model

#### Sensor-only example:

```bash
python src/train.py --model sensor \
  --csv_path dataset/Gas_Sensors_Measurements.csv \
  --save_path saved_models/sensor_model.keras \
  --epochs 30
```

#### Image-only example:

```bash
python src/train.py --model image \
  --img_path dataset/Thermal_Camera_Images \
  --save_path saved_models/image_model.keras \
  --epochs 30
```

#### Multimodal Fusion example:

```bash
python src/train.py --model multimodal \
  --csv_path dataset/Gas_Sensors_Measurements.csv \
  --img_path dataset/Thermal_Camera_Images \
  --save_path saved_models/fusion_model.keras \
  --epochs 30
```

### 4. Evaluate and visualize

```bash
python src/evaluate.py \
  --model checkpoints/fusion_model.keras \
  --mc_samples 50 \
  --output figures/uncertainty_plot.png
```

---

## 📂 Repository Structure

```plaintext
gas-detection/
├── dataset/
│   ├── Gas_Sensors_Measurements.csv     # Sensor readings
│   └── Thermal_Camera_Images/           # Images in 4 class folders
│       ├── Mixture/
│       ├── NoGas/
│       ├── Perfume/
│       └── Smoke/
│
├── src/
│   ├── data_loader.py                   # tf.data pipelines for loading sensor and image data
│   ├── models.py                        # Model definitions (sensor, image, fusion)
│   ├── train.py                         # Train models from terminal
│   ├── evaluate.py                      # Evaluate models with MC-Dropout
│
├── notebooks/
│   └── *.ipynb                          # Exploratory analysis and experimental notebooks
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📓 Notebooks

You can explore or rerun our experiments via the notebooks below:

| Notebook                                                                           | Description                                                                                         |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [abilation-multimodal.ipynb](notebooks/abilation-multimodal.ipynb)                 | Ablation experiments on the multimodal model, testing sensor dropout robustness. |
| [abilation-sensors.ipynb](notebooks/abilation-sensors.ipynb)                       | Sensor-only ablation on sensor-only model testing sensor dropout robustness.  |
| [final\_late\_fusion\_code.ipynb](notebooks/final_late_fusion_code.ipynb)          | Late fusion architecture combining sensor and image predictions at the decision level using different late fusion techniques.              |
| [gas-detection-image-mc.ipynb](notebooks/gas-detection-image-mc.ipynb)             | Image-only Image-only model with MC-Dropout enabled at inference for uncertainty estimation.                   |
| [gas-detection-image.ipynb](notebooks/gas-detection-image.ipynb)                   | Standard image-only classification model.                                                           |
| [gas-sensor-model.ipynb](notebooks/gas-sensor-model.ipynb)                         | Baseline sensor-only model development and training.                                                |
| [genetic-algorithm.ipynb](notebooks/genetic-algorithm.ipynb)                       | Genetic Algorithm for tuning model hyperparameters for sensor model.                                                 |
| [intermediate-fusion-mc-ga.ipynb](notebooks/intermediate-fusion-mc-ga.ipynb)       | Intermediate fusion with both MC-Dropout and GA-based hyperparameter tuning.                        |
| [intermediate-fusion-mc-model.ipynb](notebooks/intermediate-fusion-mc-model.ipynb) | Multimodal intermediate fusion model with MC-Dropout layers for uncertainty-aware fusion.                               |
| [intermediate-fusion.ipynb](notebooks/intermediate-fusion.ipynb)                   | Baseline intermediate fusion model.                                    |
| [sensor-eda.ipynb](notebooks/sensor-eda.ipynb)                                     | Sensor feature analysis: correlation heatmap, distributions, and sensor ablation impact.            |
| [sensors-with-ga-mc.ipynb](notebooks/sensors-with-ga-mc.ipynb)                     | Sensor model with both GA-based tuning and MC-Dropout inference.                                    |

---

## 📊 Results

We evaluate all 14 model configurations described in the experiments, spanning sensor-only, image-only, intermediate fusion, and late fusion strategies. Each model is tested on the same held-out set, with accuracy as the primary evaluation metric. Models employing Monte Carlo (MC) dropout report the mean prediction over T=50 stochastic forward passes. For configurations involving Genetic Algorithm (GA) tuning, the layer dropout rate, hidden layer widths, and learning rate were optimised using a population of 40 over 50 generations.

| Category               | Model      | MC  | GA  | Accuracy |
|------------------------|------------|-----|-----|----------|
| **Unimodal**           | S‑plain    |     |     | 0.8710   |
|                        | S‑MC       | ✓   |     | 0.8750   |
|                        | S‑MC‑GA    | ✓   | ✓   | **0.9640** |
|                        | I‑plain    |     |     | **0.9430** |
|                        | I‑MC       | ✓   |     | 0.9410   |
| **Intermediate Fusion**| M‑fusion   |     |     | 0.9840   |
|                        | M‑MC       | ✓   |     | 0.9790   |
|                        | M‑MC‑GA    | ✓   | ✓   | **0.9920** |
| **Late Fusion**        | L‑WAvg‑MC  | ✓   | ✓   | **0.9906** |
|                        | L‑Vote‑MC  | ✓   | ✓   | 0.9891   |
|                        | L‑Meta‑MC  | ✓   | ✓   | **0.9906** |
|                        | L‑WAvg     |     | ✓   | 0.9891   |
|                        | L‑Vote     |     | ✓   | 0.9891   |
|                        | L‑Meta     |     | ✓   | **0.9906** |

## 📁 Data Sources

We use the publicly available multimodal gas dataset of thermal images and sensor readings:

* **Mendeley Data**: "Multimodal Gas Sensing and Thermal Imaging Dataset" (Version 2), Mendeley Data, 2024. doi:10.17632/zkwgkjkjn9.2

Please cite the original dataset in any publications:

> T. Samal and A. Ghosh, *Multimodal Gas Sensing and Thermal Imaging Dataset* (Version 2), Mendeley Data, 2024.

---

## 🤝 Contributions

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is licensed under the [Apache License](LICENSE).
