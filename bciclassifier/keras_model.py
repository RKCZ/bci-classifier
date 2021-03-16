from tensorflow import keras
from tensorflow.keras import layers


def keras_model_target(name, optimizer='rmsprop', init='glorot_uniform'):
    # create model
    inputs = keras.Input(shape=(9824,))
    dense = layers.Dense(512, activation=keras.activations.relu, kernel_initializer=init)
    x = dense(inputs)
    x = layers.Dense(512, activation=keras.activations.relu)(x)
    outputs = layers.Dense(2, activation=keras.activations.softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    # compile model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def keras_model_audiovisual(name, optimizer='rmsprop', init='glorot_uniform'):
    # create model
    inputs = keras.Input(shape=(9824,))
    dense = layers.Dense(512, activation=keras.activations.relu, kernel_initializer=init)
    x = dense(inputs)
    x = layers.Dense(512, activation=keras.activations.relu)(x)
    outputs = layers.Dense(3, activation=keras.activations.softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    # compile model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model
