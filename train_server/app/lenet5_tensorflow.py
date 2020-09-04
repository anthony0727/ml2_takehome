import argparse
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets.mnist import load_data

def load_mnist():
    (train_X, train_Y), (test_X, test_Y) = load_data()
    train_X = train_X/255.
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = test_X/255. 
    test_X = np.expand_dims(test_X, axis=-1)

    print(train_X.shape)
    print(train_Y.shape)

    return (train_X, train_Y), (test_X, test_Y)

def build_lenet5():
    model = keras.Sequential()

    model.add(layers.ZeroPadding2D((2,2)))

    model.add(
        layers.Conv2D(
            filters=6, 
            kernel_size=(5, 5),
            activation='tanh', 
            input_shape=(32, 32, 1)
        )
    )
    model.add(layers.MaxPool2D())

    model.add(
        layers.Conv2D(
            filters=16, 
            kernel_size=(5, 5),
            activation='tanh'
        )
    )
    model.add(layers.MaxPool2D())

    model.add(layers.Flatten())

    model.add(
        layers.Dense(
            units=120, 
            activation='relu'
        )
    )
    model.add(
        layers.Dense(
            units=84, 
            activation='relu'
        )
    )
    model.add(
        layers.Dense(
            units=10, 
            activation='softmax'
        )
    )

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)

    args = parser.parse_args()

    train_data, test_data = load_mnist()

    model = build_lenet5()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     save_weights_only=True,
    #     verbose=1
    # )

    model.fit(
        *train_data,
        batch_size=args.batch_size, 
        epochs=args.epochs
        # validation_data=test_data
        # callbacks=[cp_callback]
    )

    loss, acc = model.evaluate(*test_data)

    model.save('../models/tensorflow')