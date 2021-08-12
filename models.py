import tensorflow as tf

from utils import *

model_example = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

export_model_to_json(model_example, "model_example.json")
