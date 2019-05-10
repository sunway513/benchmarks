import keras
import numpy as np
import tensorflow as tf


def get_model1():
    return keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])


def get_model2():
    return keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])


def train_model(model, use_tf_optimizer=True):
    optimizer = tf.train.AdamOptimizer() if use_tf_optimizer else keras.optimizers.Adam()

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype(np.float32)

    model.compile(optimizer=optimizer,
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels))

    print()


if __name__ == '__main__':
    print('model1:')
    train_model(get_model1(), False)
    train_model(get_model1(), True)

    print('model2:')
    train_model(get_model2(), False)
    train_model(get_model2(), True)
