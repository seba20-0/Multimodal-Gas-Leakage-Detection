import tensorflow as tf


# --------------- Sampling layer to simulate sensor faliure  ---------------------


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
    

# --------------- Monte Carlo Dropout Layers ---------------------    
class MCDropout(tf.keras.layers.Dropout):
    """
    Dropout that is active both at train *and* inference time,
    so we can sample N stochastic forward passes.
    """
    def call(self, inputs, training=None):
        # Force dropout even in inference
        return super().call(inputs, training=True)
    

class MCSpatialDropout2D(tf.keras.layers.SpatialDropout2D):
    """
    Dropout that is active both at train *and* inference time,
    so we can sample N stochastic forward passes.
    """
    def call(self, inputs, training=None):
        # Force dropout even in inference 
        return super().call(inputs, training=True)



# --------------- Sensor Model ---------------------

def build_sensor_model(
    input_dim=7,
    sensor_dropout_rate=0.3,
    layer_dropout = 0.5,
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
    x = MCDropout(layer_dropout)(x)
    x = tf.keras.layers.Dense(hidden_units[1], activation="relu")(x)
    x = MCDropout(layer_dropout)(x)
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
    conv_filters=(32, 64 , 128),
    layer_dropout = 0.5,
    dense_units=64,
    output_units=4,
    lr=1e-4
):
    """
    Image-only CNN classifier.
    """
    inp = tf.keras.Input(shape=img_shape, name="image_input")
    x = tf.keras.layers.Conv2D(conv_filters[0], 3, activation="relu", padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(conv_filters[1], 3, activation="relu", padding="same",use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(conv_filters[2], 3, activation="relu", padding="same",use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = MCSpatialDropout2D(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = MCDropout(layer_dropout)(x)
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
    img_shape=(120, 160, 3),
    input_dim=7,
    sensor_dropout_rate=0.3,
    layer_dropout=0.5,
    sensor_units=(64,32),
    img_dense=64,
    fusion_dense=64,
    output_units=4,
    lr=1e-4
):
    """
    Intermediate fusion of sensor and image branches.
    Enhanced image branch with deeper layers and normalization.
    """
    # Sensor input
    s_in = tf.keras.Input(shape=(input_dim,), name="sensor_input")
    s = RandomSensorDropout(sensor_dropout_rate, name="sensor_dropout")(s_in)
    s = tf.keras.layers.Dense(sensor_units[0], activation="relu")(s)
    s = MCDropout(layer_dropout)(s)
    s = tf.keras.layers.Dense(sensor_units[1], activation="relu")(s)
    s = MCDropout(layer_dropout)(s)

    # Image input
    i_in = tf.keras.Input(shape=img_shape, name="image_input")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu" ,use_bias=False)(i_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = MCSpatialDropout2D(0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(img_dense, activation="relu")(x)
    x = MCDropout(layer_dropout)(x)

    # Fusion
    fused = tf.keras.layers.Concatenate()([s, x])
    y = tf.keras.layers.Dense(fusion_dense, activation="relu")(fused)
    y = MCDropout(layer_dropout)(y)
    out = tf.keras.layers.Dense(output_units, activation="softmax", name="output")(y)

    # Model compile
    model = tf.keras.Model([s_in, i_in], out, name="multimodal")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
