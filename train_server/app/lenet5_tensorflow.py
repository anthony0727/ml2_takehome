import argparse
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets.mnist import load_data

(train_X, train_Y), (test_X, test_Y) = load_data()
train_X = train_X/255.
train_X = np.expand_dims(train_X,axis=-1)
test_X = test_X/255. 
test_X = np.expand_dims(test_X,axis=-1)

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

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

if __name__ == "__main__":
    
    model.fit(x, y, batch_size=)