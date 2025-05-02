import os, glob
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_split_sensor_data(
    csv_path: str,
    test_size: float = 0.1,
    val_size: float = 0.2,
    random_state: int = 42
):
    """
    Load sensor CSV, preprocess (drop columns, encode labels, scale), and split into train/val/test.

    Args:
        csv_path: Path to the Gas Sensors Measurements CSV file.
        test_size: Fraction of data reserved for final test set.
        val_size: Fraction of remaining data reserved for validation.
        random_state: Random seed for reproducibility.

    Returns:
        train_ds: tf.data.Dataset for training (features, labels).
        val_ds: tf.data.Dataset for validation.
        test_ds: tf.data.Dataset for testing.
    """
    # 1) Load and clean
    df = pd.read_csv(csv_path)
    # drop unused columns
    df = df.drop(columns=["Serial Number", "Corresponding Image Name"], errors='ignore')
    # encode labels
    df['Gas'] = df['Gas'].astype('category').cat.codes

    # 2) Extract features and labels
    feature_cols = [c for c in df.columns if c != 'Gas']
    X = df[feature_cols].values.astype('float32')
    y = df['Gas'].values.astype('int32')

    # 3) Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Split train_temp into train and val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction,
        stratify=y_temp, random_state=random_state
    )

    # 5) Build tf.data datasets
    def make_ds(features, labels, batch_size=32, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(features), seed=random_state)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    batch_size = 32
    train_ds = make_ds(X_train, y_train, batch_size, shuffle=True)
    val_ds   = make_ds(X_val,   y_val,   batch_size, shuffle=False)
    test_ds  = make_ds(X_test,  y_test,  batch_size, shuffle=False)

    return train_ds, val_ds, test_ds

def load_and_split_image_data(
    data_dir: str,
    test_size: float = 0.1,
    val_size: float = 0.2,
    batch_size: int = 32,
    img_size: tuple = (120, 160),
    random_state: int = 42
):
    """
    Load images from class-named subfolders, exclude 'Sample...' folder, preprocess, and split.

    Returns tf.data.Dataset objects for image-only data.
    """

    # find class directories
    class_dirs = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d))
                  and not d.startswith("Sample")]
    label_map = {name: idx for idx, name in enumerate(sorted(class_dirs))}

    paths, labels = [], []
    for cls in class_dirs:
        folder = os.path.join(data_dir, cls)
        imgs = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
        paths.extend(imgs)
        labels.extend([label_map[cls]] * len(imgs))
    paths = np.array(paths)
    labels = np.array(labels)

    X_temp, X_test, y_temp, y_test = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=random_state
    )

    def process_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def make_ds(paths, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=random_state)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train, y_train, shuffle=True)
    val_ds   = make_ds(X_val,   y_val,   shuffle=False)
    test_ds  = make_ds(X_test,  y_test,  shuffle=False)

    return train_ds, val_ds, test_ds

def load_and_split_multimodal_data(
    sensor_csv_path: str,
    image_data_dir: str,
    test_size: float = 0.1,
    val_size: float = 0.2,
    batch_size: int = 32,
    img_size: tuple = (120, 160),
    random_state: int = 42
):
    """
    Load and split both sensor CSV and image folders into multimodal tf.data.Datasets.

    Args:
        sensor_csv_path: Path to the gas sensor CSV.
        image_data_dir: Base directory of image class subfolders.
        test_size: Fraction of data reserved for testing.
        val_size: Fraction of remaining data reserved for validation.
        batch_size: Batch size for tf.data pipelines.
        img_size: Size to resize images to (height, width).
        random_state: Seed for reproducibility.

    Returns:
        train_ds, val_ds, test_ds: tf.data.Dataset yielding ((sensor, image), label).
    """
    # 1) Load sensor CSV
    sdf = pd.read_csv(sensor_csv_path)
    sdf = sdf.drop(columns=["Serial Number"], errors='ignore')
    sdf['Gas'] = sdf['Gas'].astype('category').cat.codes

    # Build image filename column
    sdf['Image_File'] = sdf['Corresponding Image Name'].astype(str) + ".png"

    # Extract sensor features and labels
    sensor_cols = [c for c in sdf.columns if c not in ['Gas', 'Corresponding Image Name', 'Image_File']]
    sensors = sdf[sensor_cols].values.astype('float32')
    labels = sdf['Gas'].values.astype('int32')

    # Normalize sensors
    scaler = StandardScaler().fit(sensors)
    sensors = scaler.transform(sensors)

    # Map image names to full paths
    base = image_data_dir
    def find_path(fname):
        for cls in os.listdir(base):
            p = os.path.join(base, cls, fname)
            if os.path.exists(p):
                return p
        return None

    paths = np.array(sdf['Image_File'].map(find_path))
    valid = ~pd.isna(paths)

    sensors = sensors[valid]
    paths   = paths[valid]
    labels  = labels[valid]

    # Split multimodal arrays
    X_temp, X_test, S_temp, S_test, y_temp, y_test = train_test_split(
        paths, sensors, labels,
        test_size=test_size, stratify=labels, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, S_train, S_val, y_train, y_val = train_test_split(
        X_temp, S_temp, y_temp,
        test_size=val_frac, stratify=y_temp, random_state=random_state
    )

    # Define loader
    def loader(path, sens, lab):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        sens = tf.cast(sens, tf.float32)
        return (sens, img), lab

    # Build tf.data datasets
    def make_ds(paths, sensors, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, sensors, labels))
        ds = ds.map(loader, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(labels), seed=random_state)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train, S_train, y_train, shuffle=True)
    val_ds   = make_ds(X_val,   S_val,   y_val,   shuffle=False)
    test_ds  = make_ds(X_test,  S_test,  y_test,  shuffle=False)

    return train_ds, val_ds, test_ds