#!/usr/bin/env python3
"""
Training script for sensor-only, image-only, and multimodal models using data_loader.

"""
import argparse
import tensorflow as tf
from models import build_sensor_model, build_image_model, build_multimodal_model
from data_loader import load_and_split_sensor_data, load_and_split_image_data, load_and_split_multimodal_data

def main(args):
    # Load data based on model type
    if args.model == "sensor":
        train_ds, val_ds, _ = load_and_split_sensor_data(
            args.csv_path, test_size=0.1, val_size=0.2
        )
        model = build_sensor_model(input_dim=args.input_dim)

    elif args.model == "image":
        train_ds, val_ds, _ = load_and_split_image_data(
            args.img_path, test_size=0.1, val_size=0.2
        )
        model = build_image_model()

    else:  # multimodal
        train_ds, val_ds, _ = load_and_split_multimodal_data(
            args.csv_path, args.img_path, test_size=0.1, val_size=0.2
        )
        model = build_multimodal_model()

    # Save best model based on val accuracy
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Train the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal model from terminal")
    parser.add_argument("--model", type=str, default="multimodal",
                        choices=["sensor", "image", "multimodal"], help="Which model to train")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the sensor CSV file")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image dataset")
    parser.add_argument("--save_path", type=str, default="../saved_models/best_model.keras", help="Where to save best model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--input_dim", type=int, default=7, help="Number of sensor features (only for sensor model)")

    args = parser.parse_args()
    main(args)
