import tensorflow as tf
import numpy as np
import pygad.kerasga
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def create_model(input_size, hidden_sizes, output_layer_size):
    input_layer = tf.keras.layers.Input(input_size)

    last = input_layer
    for s in hidden_sizes:
        last = tf.keras.layers.Dense(s, activation="relu")(last)
    output_layer = tf.keras.layers.Dense(output_layer_size, activation="tanh")(last)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def train_model(model, files, epochs):
    x_data = []
    y_data = []
    for file in files:
        file_data = np.genfromtxt(file, delimiter=',').tolist()
        for r in file_data:
            x_data.append(r[:17])
            y_data.append(r[17:])

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    print(x_data.shape)
    print(y_data.shape)

    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, batch_size=2048, epochs=epochs, validation_data=(X_val, y_val))

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def get_model_from_solution(model, solution):
    # Fetch the parameters of the best solution.
    solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                             weights_vector=solution)
    _model = tf.keras.models.clone_model(model)
    _model.set_weights(solution_weights)

    return _model


def predict_for_drone(model, drone):
    data_input = np.array([drone.ml])

    predictions = model(data_input)

    predictions = predictions[0]

    drone.propellers[0].force = max(-1.0, min(float(predictions[0]), 1.0))
    drone.propellers[1].force = max(-1.0, min(float(predictions[1]), 1.0))
    drone.propellers[2].force = max(-1.0, min(float(predictions[2]), 1.0))
    drone.propellers[3].force = max(-1.0, min(float(predictions[3]), 1.0))
