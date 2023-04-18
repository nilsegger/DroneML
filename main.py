import math
import pickle

from sim import DroneSimulation, paths

import numpy as np
import pygad
import pygad.gann
import pygad.nn
import random
import time
import datetime
import math

from pynput import keyboard

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = "progress_" + formatted_time + ".pkl"

initial_population_file = 'progress_2023-04-18_21-56-04.pkl' # None # 'lucky_drone_fitness_81.pkl' # None # 'test.pkl'

output_population_file = filename # 'test.pkl'
initial_population_matrices = None

render_only_best = True

step = 1 / 15

num_solutions = 20
GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=28,  # 28 inputs
                                num_neurons_hidden_layers=[13, 8, 6],
                                num_neurons_output=4,
                                hidden_activations=["relu", "relu", "relu"],
                                output_activation="sigmoid")

if initial_population_file is not None:
    try:
        file = open(initial_population_file, 'rb')
        initial_population_matrices = pickle.load(file)
        file.close()
        GANN_instance.update_population_trained_weights(population_trained_weights=initial_population_matrices)
    except FileNotFoundError:
        print(initial_population_file, "does not exist, continuing without")

population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

simulation = DroneSimulation()

if not render_only_best:
    simulation.SetupIrrlicht()


def run_simulation(sol_idx, timestep):
    data_inputs = np.array([
        simulation.DronePosition() + simulation.DroneRotation() + simulation.DroneSensors() + simulation.NextTarget()])

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_inputs,
                                   problem_type="regression")

    predictions = predictions[0]

    """
    old_predictions = predictions.copy()

    predictions[0] = max(0, min(predictions[0], 1))
    predictions[1] = max(0, min(predictions[1], 1))
    predictions[2] = max(0, min(predictions[2], 1))
    predictions[3] = max(0, min(predictions[3], 1))
    """

    simulation.propellers[0].force = predictions[0]
    simulation.propellers[1].force = predictions[1]
    simulation.propellers[2].force = predictions[2]
    simulation.propellers[3].force = predictions[3]

    # print(old_predictions, "->", predictions)

    simulation.Update(timestep)

    simulation.fitness_update(timestep)


def fitness_func(ga_instance, solution, sol_idx):
    global GANN_instance

    timeout = 10.0

    simulation.clear()
    simulation.SetupWorld(random.choice(paths))
    simulation.SetupPoints(0, 5, 50, 10, 1, 5.0, 4, 10, 0.05)

    while simulation.timer <= timeout:

        run_simulation(sol_idx, step)

        if not render_only_best and simulation.window_open():
            simulation.Render()

    simulation.fitness_final()

    # print("points:", simulation.points)
    return simulation.points


def callback_generation(ga_instance):
    global GANN_instance

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

    if ga_instance.generations_completed % 50 == 0:
        ga_instance.plot_fitness()


initial_population = population_vectors.copy()

num_parents_mating = 4

num_generations = 100

mutation_percent_genes = 10

parent_selection_type = "sus"

crossover_type = "single_point"

mutation_type = "random"

crossover_probability = 0.2

keep_parents = 1

keep_elitism = 3

init_range_low = -2
init_range_high = 5

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       on_generation=callback_generation)

ga_instance.run()

population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                        population_vectors=ga_instance.population)

file = open(output_population_file, 'wb')
pickle.dump(population_matrices, file)
file.close()

ga_instance.plot_fitness()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

keys = {
    'r': False
}


def on_press(key):
    global keys

    try:
        keys[key.char] = True
    except AttributeError:
        keys[key] = True


def on_release(key):
    global keys
    try:
        keys[key.char] = False
    except AttributeError:
        keys[key] = False


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)

listener.start()

if render_only_best:
    simulation.SetupIrrlicht()

simulation.clear()
simulation.SetupWorld(random.choice(paths))

simulation.SetupPoints(0, 0.1, 15, 80, 10, 20.0, 3, 10, math.sqrt(5))

last = time.time()

while simulation.window_open():
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - last
    last = time.time()

    run_simulation(solution_idx, elapsed_time)

    simulation.Render()

    if keys['r'] or simulation.timer >= 10.0:
        simulation.clear()

        simulation.SetupWorld(random.choice(paths))

        simulation.SetupPoints(0, 0.1, 15, 80, 10, 20.0, 3, 10, math.sqrt(5))
