import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization,Reshape,Conv2DTranspose,Flatten,Conv2D,Dense,Dropout

class Descriminator(tf.keras.Model):

    def __init__(self):
        super(Descriminator, self).__init__()
        self._conv1 = Conv2D(filters=64, kernel_size=5, strides=2, padding="same")
        self._drop1 = Dropout(0.3)
        self._conv2 = Conv2D(filters=128, kernel_size=5, strides=2, padding="same")
        self._drop2 = Dropout(0.3)
        self._flatten = Flatten()
        self._dense = Dense(1)

    def call(self, x, training):
        x = self._conv1(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self._drop1(x,training=training)

        x = self._conv2(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self._drop2(x,training=training)

        x = self._flatten(x, training=training)
        x = self._dense(x, training=training)
        return x


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self._dense = Dense(7*7*8)
        self._bn1 = BatchNormalization()
        self._convt1 = Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding="same")
        self._bn2 = BatchNormalization()
        self._convt2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding="same")
        self._bn3 = BatchNormalization()
        self._out = Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding="same", activation="tanh")

    def call(self, x, training):
        x = self._dense(x, training=training)
        x = self._bn1(x, training= training)
        x = tf.nn.leaky_relu(x)
        x = tf.reshape(x, (64,7,7,8))

        x = self._convt1(x, training=training)
        x = self._bn2(x, training= training)
        x = tf.nn.leaky_relu(x)

        x = self._convt2(x, training=training)
        x = self._bn3(x, training= training)
        x = tf.nn.leaky_relu(x)

        x = self._out(x, training=training)

        return x
