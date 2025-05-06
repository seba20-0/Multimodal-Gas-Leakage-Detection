"""
python src/evaluate.py \
  --model multimodal \
  --csv_path dataset/Gas_Sensors_Measurements.csv \
  --img_path dataset/Thermal_Camera_Images \
  --model_path checkpoints/fusion_model.keras \
  --mc_samples 50 \
  --output_plot figures/fusion_uncertainty.png
  
"""


import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data_loader import (
    load_and_split_sensor_data,
    load_and_split_image_data,
    load_and_split_multimodal_data
)

def predict_mc(model, dataset, mc_samples=50):
    all_preds, all_labels = [], []

    for (x, y) in dataset:
        batch_preds = []
        for _ in range(mc_samples):
            preds = model(x, training=True).numpy()
            batch_preds.append(preds)
        batch_preds = np.stack(batch_preds, axis=0)  # (T, B, C)
        mean_preds = np.mean(batch_preds, axis=0)    # (B, C)
        all_preds.append(mean_preds)
        all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_preds, all_labels


def plot_uncertainty(preds, labels, output_path):
    entropy = -np.sum(preds * np.log(preds + 1e-10), axis=1)
    confidence = np.max(preds, axis=1)
    correct = np.argmax(preds, axis=1) == labels

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(entropy[correct], bins=30, alpha=0.6, label='Correct')
    plt.hist(entropy[~correct], bins=30, alpha=0.6, label='Incorrect')
    plt.xlabel("Predictive Entropy")
    plt.ylabel("Count")
    plt.title("Uncertainty Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(confidence, entropy, c=correct, cmap="coolwarm", s=10)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Entropy")
    plt.title("Confidence vs Entropy")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Uncertainty plot saved to: {output_path}")


def main(args):
    # Load data
    if args.model == "sensor":
        _, _, test_ds = load_and_split_sensor_data(args.csv_path)
    elif args.model == "image":
        _, _, test_ds = load_and_split_image_data(args.img_path)
    else:
        _, _, test_ds = load_and_split_multimodal_data(args.csv_path, args.img_path)

    # Load model
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"RandomSensorDropout": tf.keras.layers.Layer,
                        "MCDropout": tf.keras.layers.Dropout,
                        "MCSpatialDropout2D": tf.keras.layers.SpatialDropout2D}
    )

    # Predict
    preds, labels = predict_mc(model, test_ds, mc_samples=args.mc_samples)
    acc = np.mean(np.argmax(preds, axis=1) == labels)
    print(f"Test Accuracy: {acc:.4f}")

    # Save plot
    if args.output_plot:
        plot_uncertainty(preds, labels, args.output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with MC-Dropout")
    parser.add_argument("--model", type=str, default="multimodal",
                        choices=["sensor", "image", "multimodal"])
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=False)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mc_samples", type=int, default=50)
    parser.add_argument("--output_plot", type=str, default="uncertainty_plot.png")

    args = parser.parse_args()
    main(args)
