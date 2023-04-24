import datetime
import time
import random

import pygad.kerasga
import tensorflow as tf
import numpy as np

from sim import Simulation, paths, make_path_dense, PointsConfig

# Hide GPU from visible devices
# from what we gather, gpu is only really an improvement if you are teaching your neural network
tf.config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

input_size = 35
hiddenlayer1_size = 256
hiddenlayer2_size = 256
output_layer_size = 4

step = 1 / 30
points_config = PointsConfig(5, 10, 50, 1, 5, 2, 5, 0.01, 0.1)

num_solutions = 10
num_generations = 10
num_parents_mating = 4

input_layer = tf.keras.layers.Input(input_size)
dense_layer1 = tf.keras.layers.Dense(hiddenlayer1_size, activation="relu")(input_layer)
dense_layer2 = tf.keras.layers.Dense(hiddenlayer2_size, activation="relu")(input_layer)
output_layer = tf.keras.layers.Dense(output_layer_size, activation="tanh")(dense_layer2)

keras_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

keras_ga = pygad.kerasga.KerasGA(model=keras_model,
                                 num_solutions=10)

make_path_dense(paths, 0.1)


def get_model_from_solution(model, solution):
    # Fetch the parameters of the best solution.
    solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                             weights_vector=solution)
    _model = tf.keras.models.clone_model(model)
    _model.set_weights(solution_weights)

    return _model


def fitness_func(instance, solution, solution_idx):
    global keras_ga, keras_model

    simulation = Simulation(points_config, update_rays_every_n_frame=10)

    begin = time.time()

    timeout = 15.0

    simulation.clear()
    simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)

    weighted_model = get_model_from_solution(keras_model, solution)

    while simulation.timer <= timeout:
        data_input = np.array([simulation.drones[0].ml])

        predictions = weighted_model(data_input)

        predictions = predictions[0]

        simulation.drones[0].propellers[0].force = max(-1.0, min(float(predictions[0]), 1.0))
        simulation.drones[0].propellers[1].force = max(-1.0, min(float(predictions[1]), 1.0))
        simulation.drones[0].propellers[2].force = max(-1.0, min(float(predictions[2]), 1.0))
        simulation.drones[0].propellers[3].force = max(-1.0, min(float(predictions[3]), 1.0))

        simulation.update(step)

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - begin

    points = [drone.points for drone in simulation.drones]

    print(1, "networks", round(elapsed_time), "seconds", points)

    return points[0]


def callback_generation(instance):
    print("Generation = {generation}".format(generation=instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=instance.best_solution()[1]))


initial_population = keras_ga.population_weights  # Initial population of network weights

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    initial_population=initial_population,
    fitness_func=fitness_func,
    on_generation=callback_generation
)

ga_instance.run()

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = "progress_" + formatted_time + ".pkl"
ga_instance.fitness_func = None
ga_instance.save(filename)
ga_instance.fitness_func = fitness_func

ga_instance.plot_fitness("Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))

"""
# Make prediction based on the best solution.
predictions = pygad.kerasga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs)
print("Predictions : \n", predictions)

mae = tf.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print("Absolute Error : ", abs_error)
"""
