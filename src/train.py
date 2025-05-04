#!/usr/bin/env python3
"""
Training script for sensor-only, image-only, and multimodal models using data_loader.

Usage examples:
  python train.py sensor \
    --sensor-data data/sensor.csv \
    --output checkpoints/sensor_model \
    --epochs 30 \
    --sensor-dropout 0.3 \
    --layer-dropout 0.5 \
    --hidden-units 64 32

  python train.py image \
    --image-data-dir data/images \
    --output checkpoints/image_model \
    --epochs 25 \
    --layer-dropout 0.5 \
    --dense-units 128 \
    --conv-filters 32 64

  python train.py fusion \
    --sensor-data data/sensor.csv \
    --image-data-dir data/images \
    --output checkpoints/fusion_model \
    --epochs 20 \
    --sensor-dropout 0.3 \
    --layer-dropout 0.5 \
    --sensor-units 32 \
    --img-dense 64 \
    --fusion-dense 128
"""
import argparse
import tensorflow as tf

from data_loader import (
    load_and_split_sensor_data,
    load_and_split_image_data,
    load_and_split_multimodal_data
)
from models import (
    build_sensor_model,
    build_image_model,
    build_multimodal_model
)


def train_sensor(args):
    train_ds, val_ds, _ = load_and_split_sensor_data(
        args.sensor_data,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

    model = build_sensor_model(
        input_dim=None,  # inferred by model.fit
        sensor_dropout_rate=args.sensor_dropout,
        layer_dropout=args.layer_dropout,
        hidden_units=tuple(args.hidden_units),
        output_units=args.output_units,
        lr=args.lr
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )
    model.save(args.output)
    print(f"Sensor model saved to {args.output}")


def train_image(args):
    train_ds, val_ds, _ = load_and_split_image_data(
        args.image_data_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        random_state=args.random_state
    )

    model = build_image_model(
        img_shape=tuple(args.img_size) + (3,),
        conv_filters=tuple(args.conv_filters),
        layer_dropout=args.layer_dropout,
        dense_units=args.dense_units,
        output_units=args.output_units,
        lr=args.lr
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )
    model.save(args.output)
    print(f"Image model saved to {args.output}")


def train_fusion(args):
    train_ds, val_ds, _ = load_and_split_multimodal_data(
        args.sensor_data,
        args.image_data_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        random_state=args.random_state
    )

    model = build_multimodal_model(
        img_shape=tuple(args.img_size) + (3,),
        input_dim=None,  # inferred
        sensor_dropout_rate=args.sensor_dropout,
        layer_dropout=args.layer_dropout,
        sensor_units=args.sensor_units,
        img_dense=args.img_dense,
        fusion_dense=args.fusion_dense,
        output_units=args.output_units,
        lr=args.lr
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )
    model.save(args.output)
    print(f"Multimodal model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Train sensor-only, image-only, or multimodal models"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Sensor
    ps = subparsers.add_parser("sensor", help="Train sensor-only model")
    ps.add_argument("--sensor-data", required=True, help="Path to sensor CSV")
    ps.add_argument("--output", required=True, help="Save path for sensor model")
    ps.add_argument("--epochs", type=int, default=20)
    ps.add_argument("--batch-size", type=int, default=32)
    ps.add_argument("--lr", type=float, default=1e-4)
    ps.add_argument("--sensor-dropout", type=float, default=0.3,
                    help="Dropout rate for sensor branch")
    ps.add_argument("--layer-dropout", type=float, default=0.5,
                    help="Dropout rate for sensor dense layers")
    ps.add_argument("--hidden-units", type=int, nargs=2, default=[64,32],
                    help="Number of units in each hidden sensor dense layer")
    ps.add_argument("--output-units", type=int, default=4)
    ps.add_argument("--test-size", type=float, default=0.1)
    ps.add_argument("--val-size", type=float, default=0.2)
    ps.add_argument("--random-state", type=int, default=42)

    # Image
    pi = subparsers.add_parser("image", help="Train image-only model")
    pi.add_argument("--image-data-dir", required=True,
                    help="Root directory of image classes")
    pi.add_argument("--output", required=True, help="Save path for image model")
    pi.add_argument("--epochs", type=int, default=20)
    pi.add_argument("--batch-size", type=int, default=32)
    pi.add_argument("--lr", type=float, default=1e-4)
    pi.add_argument("--conv-filters", type=int, nargs=2, default=[32,64],
                    help="Filters in each Conv2D layer")
    pi.add_argument("--dense-units", type=int, default=64,
                    help="Units in dense layer before dropout")
    pi.add_argument("--output-units", type=int, default=4)
    pi.add_argument("--img-size", type=int, nargs=2, default=[120,160])
    pi.add_argument("--layer-dropout", type=float, default=0.5,
                    help="Dropout rate for image dense layer")
    pi.add_argument("--test-size", type=float, default=0.1)
    pi.add_argument("--val-size", type=float, default=0.2)
    pi.add_argument("--random-state", type=int, default=42)

    # Fusion
    pf = subparsers.add_parser("fusion", help="Train multimodal model")
    pf.add_argument("--sensor-data", required=True, help="Path to sensor CSV")
    pf.add_argument("--image-data-dir", required=True,
                    help="Root directory of image classes")
    pf.add_argument("--output", required=True, help="Save path for fusion model")
    pf.add_argument("--epochs", type=int, default=20)
    pf.add_argument("--batch-size", type=int, default=32)
    pf.add_argument("--lr", type=float, default=1e-4)
    pf.add_argument("--sensor-dropout", type=float, default=0.3,
                    help="Dropout rate for sensor branch")
    pf.add_argument("--layer-dropout", type=float, default=0.5,
                    help="Dropout rate for all dense layers in fusion")
    pf.add_argument("--sensor-units", type=int, default=32,
                    help="Units in sensor branch dense layer")
    pf.add_argument("--img-dense", type=int, default=64,
                    help="Units in image branch dense layer")
    pf.add_argument("--fusion-dense", type=int, default=64,
                    help="Units in fusion dense layer")
    pf.add_argument("--img-size", type=int, nargs=2, default=[120,160])
    pf.add_argument("--test-size", type=float, default=0.1)
    pf.add_argument("--val-size", type=float, default=0.2)
    pf.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    if args.mode == "sensor":
        train_sensor(args)
    elif args.mode == "image":
        train_image(args)
    elif args.mode == "fusion":
        train_fusion(args)
    else:
        parser.error("Choose 'sensor', 'image', or 'fusion'.")


if __name__ == "__main__":
    main()
