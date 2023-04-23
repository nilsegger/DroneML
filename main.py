from keras import Sequential, Input
from keras.layers import Dense

from sim import Simulation, SimulationVisual, paths, make_path_dense, PointsConfig

import numpy as np
import pygad.nn
import random
import datetime
import time

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = "progress_" + formatted_time + ".pkl"

initial_population_file = None  # 'progress_2023-04-18_22-05-33.pkl' # None # 'lucky_drone_fitness_81.pkl' # None # 'test.pkl'

output_population_file = filename  # 'test.pkl'
initial_population_matrices = None

render_only_best = True

make_path_dense(paths, 0.1)

step = 1 / 30
points_config = PointsConfig(5, 10, 50, 1, 5, 2, 5, 0.01, 0.1)
simulation = Simulation(points_config, update_rays_every_n_frame=6)
visualisation = None

if not render_only_best:
    visualisation = SimulationVisual(simulation)


def fitness_func(ga_instance, solution, solution_idx):

    # Create a neural network using the solution weights and biases
    network = create_network(solution)

    # Run the simulation and calculate the fitness score
    fitness = run_simulation(network, solution_idx)

    # Return the fitness score
    return fitness


def run_simulation(network, solution_idx):
    # Run the simulation with the given neural network
    # e.g. use a physics engine to simulate the drone movement
    # evaluate the fitness based on the problem definition

    begin = time.time()

    timeout = 30.0

    simulation.clear()
    simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)

    if not render_only_best:
        visualisation.bind_from_simulation()

    while simulation.timer <= timeout:

        data_inputs = np.array([simulation.drones[0].ml])

        # print("Inputs", data_inputs[0])

        predictions = network(data_inputs, training=False)

        simulation.drones[0].propellers[0].force = (float(predictions[0][0]) + 1)/2
        simulation.drones[0].propellers[1].force = (float(predictions[0][1]) + 1)/2
        simulation.drones[0].propellers[2].force = (float(predictions[0][2]) + 1)/2
        simulation.drones[0].propellers[3].force = (float(predictions[0][3]) + 1)/2

        # print("Predictions:", predictions)

        simulation.update(step)

        if not render_only_best and visualisation.is_window_open():
            visualisation.render()

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - begin

    print("One simulation took:", elapsed_time)

    return simulation.drones[0].points  # [drone.points for drone in simulation.drones]


def create_network(solution):
    # Create a neural network with 25 input nodes and 4 output nodes
    inputsize = 28
    hiddenlayer1 = 256
    hiddenlayer2 = 256
    outputlayer = 4

    network = Sequential([
        Input(shape=(inputsize,)),
        Dense(hiddenlayer1, activation='tanh', use_bias=True),
        Dense(hiddenlayer2, activation='tanh', use_bias=True),
        Dense(outputlayer, activation='relu', use_bias=True)
    ])

    # Set the weights and biases of the neural network to the values in the solution array
    start = 0
    # Get the shape of the layer's weights
    temp1 = inputsize * hiddenlayer1
    temp2 = temp1 + hiddenlayer1
    temp3 = temp2 + hiddenlayer1 * hiddenlayer2
    temp4 = temp3 + hiddenlayer2
    temp5 = temp4 + hiddenlayer2 * outputlayer

    weights_layer1 = solution[:temp1].reshape((inputsize, hiddenlayer1))
    biases_layer1 = solution[temp1:temp2]
    weights_layer2 = solution[temp2:temp3].reshape((hiddenlayer1, hiddenlayer2))
    biases_layer2 = solution[temp3:temp4]
    weights_layer3 = solution[temp4:temp5].reshape((hiddenlayer2, outputlayer))
    biases_layer3 = solution[temp5:]

    # Set the weights and biases of each layer
    network.layers[0].set_weights([weights_layer1, biases_layer1])
    network.layers[1].set_weights([weights_layer2, biases_layer2])
    network.layers[2].set_weights([weights_layer3, biases_layer3])

    return network


# initial_population = population_vectors.copy()

num_parents_mating = 4

num_generations = 1

mutation_percent_genes = 10

sol_per_pop = 100

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
                       # initial_population=initial_population,
                       sol_per_pop=sol_per_pop,
                       num_genes=74244,
                       fitness_func=fitness_func,
                       fitness_batch_size=20,
                       mutation_percent_genes=mutation_percent_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       # keep_parents=keep_parents,
                       # keep_elitism=keep_elitism,
                       # on_generation=callback_generation,
                       # parallel_processing=10
                       )

ga_instance.run()
