import tensorflow as tf
import numpy as np
import pygad.kerasga


def create_model(input_size, hiddenlayer1_size, hiddenlayer2_size, output_layer_size):
    input_layer = tf.keras.layers.Input(input_size)
    dense_layer1 = tf.keras.layers.Dense(hiddenlayer1_size, activation="relu")(input_layer)
    dense_layer2 = tf.keras.layers.Dense(hiddenlayer2_size, activation="relu")(dense_layer1)
    output_layer = tf.keras.layers.Dense(output_layer_size, activation="tanh")(dense_layer2)

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
