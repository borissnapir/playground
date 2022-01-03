import tensorflow as tf


class Classifier():
    def __init__(self):
        self.build_model()


    def build_model(self):
        model_seq = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', input_shape=(64,64,3)),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(100, activation='softmax'), # 00-99
        ])
        first_digit_output = tf.keras.layers.Dense(10, activation="softmax")(model_seq.output)
        second_digit_output = tf.keras.layers.Dense(10, activation="softmax")(model_seq.output)
        model = tf.keras.Model(inputs=model_seq.input, outputs=[first_digit_output, second_digit_output])
        self.model = model