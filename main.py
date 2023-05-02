import datetime
import time
import random
import pickle

import numpy as np

from sim import Simulation, paths, make_path_dense, PointsConfig
from train import *

load_ga_instance_file = None  # if none no file will be loaded, no extension!
save_every_n_gens = 10  # if none only final result is saved

input_size = 17
hidden_layer_sizes = [11]
output_layer_size = 4

step = 1 / 24
timeout_per_simulation = 5.0

batch_size = 10
points_config = PointsConfig(100, 0.2, 10, 1000, 100, 20, 10, 1, 5, 500, 100)

num_solutions = 20
num_generations = 100
num_parents_mating = 4

parent_selection_type = "sus"
keep_elitism = 3

crossover_type = "single_point"

mutation_type = "adaptive"
mutation_probability = (0.8, 0.05)

allow_duplicate_genes = False



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

    keras_ga = pygad.kerasga.KerasGA(model=keras_model,
                                     num_solutions=num_solutions)

    make_path_dense(paths, 0.2)

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
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_probability=mutation_probability,
            keep_elitism=keep_elitism,
            # mutation_num_genes=mutation_num_genes,
            # mutation_percent_genes=mutation_percent_genes,
            allow_duplicate_genes=allow_duplicate_genes

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
