import datetime
import keras
import time
import random
import pickle
import pychrono.core as chrono
from pygad.kerasga import model_weights_as_vector
import copy

import numpy

from sim import Simulation, make_path_dense, PointsConfig
from train import *

paths = [
    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(0, 5, 0),
        chrono.ChVectorD(0, 3, 0),
        chrono.ChVectorD(0, 5, 0),
        chrono.ChVectorD(0, 0, 0)
    ]
]

initial_population_model_save = 'pretrained_models/model_64_128_64_2023-05-04_12-42-01'

load_ga_instance_file = None  # if none no file will be loaded, no extension!
save_every_n_gens = 10  # if none only final result is saved

input_size = 17
hidden_layer_sizes = [64, 128, 64]
output_layer_size = 4

step = 1 / 30
timeout_per_simulation = 10

batch_size = 10
points_config = PointsConfig(100, 0.2, 10, 1000, 100, 20, 10, 1, 5, 500, 100)

num_solutions = 20
num_generations = 10
num_parents_mating = 4

parent_selection_type = "sus"
keep_elitism = 2


class CustomKerasGA:

    def __init__(self, model, num_solutions):
        """
        Creates an instance of the KerasGA class to build a population of model parameters.

        model: A Keras model class.
        num_solutions: Number of solutions in the population. Each solution has different model parameters.
        """

        self.model = model

        self.num_solutions = num_solutions

        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

    def create_population(self):
        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). Each element in the list holds a different weights of the Keras model.

        The method returns a list holding the weights of all solutions.
        """

        model_weights_vector = model_weights_as_vector(model=self.model)

        net_population_weights = []
        net_population_weights.append(model_weights_vector)

        for idx in range(self.num_solutions - 1):
            net_weights = copy.deepcopy(model_weights_vector)
            net_weights = numpy.array(net_weights) # + numpy.random.uniform(low=-1.0, high=1.0,
                                                                          # size=model_weights_vector.size)

            # Appending the weights to the population.
            net_population_weights.append(net_weights)

        return net_population_weights


def get_time_formatted():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def save():
    global ga_instance
    filename = "output/progress_gen_" + str(ga_instance.generations_completed) + "_" + get_time_formatted()

    ga_instance.fitness_func = None
    ga_instance.on_generation = None
    ga_instance.save(filename)
    ga_instance.fitness_func = fitness_func
    ga_instance.on_generation = callback_generation

    best_solution, _, _ = ga_instance.best_solution()
    best_solution_filename = "output/best_solution_gen_" + str(
        ga_instance.generations_completed) + "_" + get_time_formatted() + ".pkl"
    with open(best_solution_filename, 'wb') as f:
        pickle.dump(best_solution, f)


def fitness_func(instance, solutions, solution_indices):
    global keras_ga, keras_model

    if not isinstance(solution_indices, np.ndarray):
        solutions = [solutions]

    simulation.clear()
    simulation.setup_world(random.choice(paths), len(solutions), ignore_visualisation_objects=True)

    begin = time.time()

    timeout = timeout_per_simulation

    models = [get_model_from_solution(keras_model, solution) for solution in solutions]

    while simulation.timer <= timeout:

        for i in range(len(solutions)):
            predict_for_drone(models[i], simulation.drones[i])

        simulation.update(step)

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - begin

    points = [drone.points for drone in simulation.drones]

    print(len(solutions), "networks", round(elapsed_time, 3), "seconds", points)

    if isinstance(solution_indices, np.ndarray):
        return points
    return points[0]


def callback_generation(instance):
    if save_every_n_gens is not None and instance.generations_completed % save_every_n_gens == 0:
        save()

    print("Generation = {generation}".format(generation=instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=instance.best_solution()[1]))


if __name__ == '__main__':

    # Hide GPU from visible devices
    # from what we gather, gpu is only really an improvement if you are teaching your neural network
    tf.config.set_visible_devices([], 'GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # tf.debugging.set_log_device_placement(True)

    simulation = Simulation(points_config)

    keras_model = create_model(input_size, hidden_layer_sizes, output_layer_size)

    """
    train_model(keras_model,
                ['training/training1.csv', 'training/training2.csv', 'training/training3.csv', 'training/training4.csv',
                 'training/training5.csv'])
    """

    if initial_population_model_save is not None:
        keras_model = keras.models.load_model(initial_population_model_save)

    keras_ga = CustomKerasGA(model=keras_model,
                                     num_solutions=num_solutions)

    # make_path_dense(paths, 0.2)

    initial_population = keras_ga.population_weights  # Initial population of network weights



    ga_instance = None

    if load_ga_instance_file is None:
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            initial_population=initial_population,
            fitness_func=fitness_func,
            fitness_batch_size=batch_size,
            on_generation=callback_generation,
            parent_selection_type=parent_selection_type,
            keep_elitism=keep_elitism,
        )
    else:
        ga_instance = pygad.load(load_ga_instance_file)
        ga_instance.on_generation = callback_generation
        ga_instance.fitness_func = fitness_func
        ga_instance.num_generations = num_generations
        ga_instance.num_parents_mating = num_parents_mating
        ga_instance.fitness_batch_size = batch_size

    ga_instance.run()

    save()

    ga_instance.plot_fitness("Iteration vs. Fitness", linewidth=4)
