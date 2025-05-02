import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import layers as tfp_layers

class RandomSensorDropout(tf.keras.layers.Layer):
    """
    Layer that randomly zeros individual sensor channels with a given rate during training.
    Supports proper serialization.
    """
    def __init__(self, rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=False):
        if training and self.rate > 0.0:
            mask = tf.cast(tf.random.uniform(tf.shape(inputs)) > self.rate, inputs.dtype)
            return inputs * mask
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config
    

# --------------- Sensor Model ---------------------

def build_sensor_model(
    input_dim=7,
    sensor_dropout_rate=0.3,
    hidden_units=(64, 32),
    output_units=4,
    lr=1e-4
):
    """
    Sensor-only MLP with RandomSensorDropout option for ablation.
    """
    inp = tf.keras.Input(shape=(input_dim,), name="sensor_input")
    x = RandomSensorDropout(sensor_dropout_rate, name="sensor_dropout")(inp)
    x = tf.keras.layers.Dense(hidden_units[0], activation="relu")(x)
    x = tf.keras.layers.Dropout(sensor_dropout_rate)(x)
    x = tf.keras.layers.Dense(hidden_units[1], activation="relu")(x)
    x = tf.keras.layers.Dropout(sensor_dropout_rate)(x)
    out = tf.keras.layers.Dense(output_units, activation="softmax", name="output")(x)

    model = tf.keras.Model(inp, out, name="sensor_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# --------------- Image Model ---------------------

def build_image_model(
    img_shape=(120,160,3),
    conv_filters=(32, 64),
    dense_units=64,
    output_units=4,
    lr=1e-4
):
    """
    Image-only CNN classifier.
    """
    inp = tf.keras.Input(shape=img_shape, name="image_input")
    x = tf.keras.layers.Conv2D(conv_filters[0], 3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(conv_filters[1], 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(output_units, activation="softmax", name="output")(x)

    model = tf.keras.Model(inp, out, name="image_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model



# --------------- Multimodal Model ---------------------
def build_multimodal_model(
    img_shape=(120,160,3),
    input_dim=7,
    sensor_dropout_rate=0.3,
    sensor_units=32,
    img_dense=64,
    fusion_dense=64,
    output_units=4,
    lr=1e-4
):
    """
    Intermediate fusion of sensor and image branches.
    """
    s_in = tf.keras.Input(shape=(input_dim,), name="sensor_input")
    i_in = tf.keras.Input(shape=img_shape, name="image_input")

    # Sensor branch
    s = RandomSensorDropout(sensor_dropout_rate, name="sensor_dropout")(s_in)
    s = tf.keras.layers.Dense(sensor_units, activation="relu")(s)

    # Image branch
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(i_in)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(img_dense, activation="relu")(x)

    # Fusion
    fused = tf.keras.layers.Concatenate()([s, x])
    y = tf.keras.layers.Dense(fusion_dense, activation="relu")(fused)
    y = tf.keras.layers.Dropout(0.5)(y)
    out = tf.keras.layers.Dense(output_units, activation="softmax", name="output")(y)

    model = tf.keras.Model([s_in, i_in], out, name="multimodal")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# --------------- Sensor Bayasian Model ---------------------
def build_bayesian_sensor_model(
    input_dim=7,
    kl_weight=1e-4,
    hidden_units=(64, 32),
    output_units=4,
    lr=1e-4
):
    """
    Sensor-only Bayesian neural network using DenseVariational layers.

    Args:
      input_dim: Number of sensor features.
      kl_weight: Weight for KL divergence loss term.
      hidden_units: Tuple of two hidden layer sizes.
      output_units: Number of classes.
      lr: Learning rate.

    Returns:
      Compiled tf.keras.Model with variational layers.
    """
    tfd = tfp.distributions
    # Default posterior and prior functions
    posterior_fn = tfp_layers.default_mean_field_normal_fn()
    prior_fn = tfp_layers.default_multivariate_normal_fn

    inp = tf.keras.Input(shape=(input_dim,), name="sensor_input")
    x = tfp_layers.DenseVariational(
        units=hidden_units[0],
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation="relu",
        name="variational_dense_1"
    )(inp)
    x = tfp_layers.DenseVariational(
        units=hidden_units[1],
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation="relu",
        name="variational_dense_2"
    )(x)
    out = tfp_layers.DenseVariational(
        units=output_units,
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation="softmax",
        name="variational_output"
    )(x)

    model = tf.keras.Model(inp, out, name="bayesian_sensor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# --------------- Image Bayasian Model ---------------------
def build_bayesian_image_model(
    img_shape=(120,160,3),
    kl_weight=1e-4,
    conv_filters=(32,64),
    dense_units=64,
    output_units=4,
    lr=1e-4
):
    """
    Image-only Bayesian CNN using DenseVariational on top of feature extractor.

    Args:
      img_shape: Input image dimensions.
      kl_weight: Weight for KL divergence term.
      conv_filters: Tuple for two Conv2D layers.
      dense_units: Units in the variational dense layer.
      output_units: Number of classes.
      lr: Learning rate.

    Returns:
      Compiled tf.keras.Model with Bayesian dense layer.
    """
    posterior_fn = tfp_layers.default_mean_field_normal_fn()
    prior_fn = tfp_layers.default_multivariate_normal_fn

    inp = tf.keras.Input(shape=img_shape, name="image_input")
    x = tf.keras.layers.Conv2D(conv_filters[0], 3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(conv_filters[1], 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)

    # Bayesian output layer
    out = tfp_layers.DenseVariational(
        units=output_units,
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation="softmax",
        name="variational_output_img"
    )(x)

    model = tf.keras.Model(inp, out, name="bayesian_image")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# --------------- Multimodal Bayasian Model ---------------------
def build_bayesian_multimodal_model(
    sensor_input_dim=7,
    img_shape=(120,160,3),
    kl_weight=1e-4,
    sensor_units=32,
    img_dense_units=64,
    fusion_units=64,
    output_units=4,
    lr=1e-4
):
    """
    Bayesian multimodal network combining sensor and image branches with variational layers.

    Args:
      sensor_input_dim: Number of sensor features.
      img_shape: Input image dimensions.
      kl_weight: Weight for KL divergence term.
      sensor_units: Units in variational sensor dense layer.
      img_dense_units: Units in image dense layer before fusion.
      fusion_units: Units in variational fusion layer.
      output_units: Number of classes.
      lr: Learning rate.

    Returns:
      Compiled tf.keras.Model.
    """
    # posterior & prior for variational layers
    posterior_fn = tfp_layers.default_mean_field_normal_fn()
    prior_fn = tfp_layers.default_multivariate_normal_fn

    # Inputs
    s_in = tf.keras.Input(shape=(sensor_input_dim,), name="sensor_input")
    i_in = tf.keras.Input(shape=img_shape, name="image_input")

    # Sensor branch variational
    s = RandomSensorDropout(0.3, name="sensor_dropout")(s_in)
    s = tfp_layers.DenseVariational(
        units=sensor_units,
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation="relu",
        name="variational_sensor"
    )(s)

    # Image branch feature extractor
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(i_in)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(img_dense_units, activation="relu", name="image_dense")(x)

    # Fusion
    fused = tf.keras.layers.Concatenate(name="fusion_concat")([s, x])
    fused = tfp_layers.DenseVariational(
        units=fusion_units,
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation="relu",
        name="variational_fusion"
    )(fused)

    # Output variational layer
    out = tfp_layers.DenseVariational(
        units=output_units,
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation="softmax",
        name="variational_output_mm"
    )(fused)

    model = tf.keras.Model([s_in, i_in], out, name="bayesian_multimodal")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

