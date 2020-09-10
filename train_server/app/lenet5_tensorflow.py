from datetime import date
import sys
import os
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

    return (train_X, train_Y), (test_X, test_Y)

def build_lenet5():
    model = keras.Sequential()

    model.add(layers.ZeroPadding2D((2,2)))

    model.add(
        layers.Conv2D(
            filters=6, 
            kernel_size=(5, 5),
            activation='relu',
            padding='same',
            input_shape=(1, 32, 32, 1)
        )
    )
    model.add(layers.MaxPool2D())

    model.add(
        layers.Conv2D(
            filters=16, 
            kernel_size=(5, 5),
            padding='same',
            activation='relu'
        )
    )
    model.add(layers.MaxPool2D())


    model.add(
        layers.Conv2D(
            filters=120, 
            kernel_size=(5, 5),
            padding='same',
            activation='relu'
        )
    )

    model.add(layers.Flatten())

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
    parser.add_argument('--model_path', type=str)

    if len(sys.argv) != 9:
        print("usage: python3 lenet5_tensorflow.py --epochs 10 --batch_size 64 --lr 0.01 --model_path ../models")
        exit()

    args = parser.parse_args()

    train_data, test_data = load_mnist()

    model = build_lenet5()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(args.lr),
        metrics=['accuracy']
    )

    model.fit(
        *train_data,
        batch_size=args.batch_size, 
        epochs=args.epochs
    )

    loss, acc = model.evaluate(*test_data)

    export_dir = os.path.join(args.model_path, 'tensorflow', date.today().strftime('%Y%m%d'))

    model.save(export_dir)

