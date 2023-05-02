import tensorflow as tf
import numpy as np
import pygad.kerasga


def create_model(input_size, hidden_sizes, output_layer_size):
    input_layer = tf.keras.layers.Input(input_size)

    last = input_layer
    for s in hidden_sizes:
        last = tf.keras.layers.Dense(s, activation="relu")(last)
    output_layer = tf.keras.layers.Dense(output_layer_size, activation="tanh")(last)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


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
