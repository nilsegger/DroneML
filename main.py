from keras import Sequential, Input
from keras.layers import Dense

from sim import Simulation, SimulationVisual, paths, make_path_dense, PointsConfig

import numpy as np
import pygad.nn
import random
import datetime
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

pool_size = 10

import tensorflow as tf

# Hide GPU from visible devices
# from what we gather, gpu is only really an improvement if you are teaching your neural network
tf.config.set_visible_devices([], 'GPU')

inputsize = 35
hiddenlayer1 = 256
hiddenlayer2 = 256
outputlayer = 4

temp1 = inputsize * hiddenlayer1
temp2 = temp1 + hiddenlayer1
temp3 = temp2 + hiddenlayer1 * hiddenlayer2
temp4 = temp3 + hiddenlayer2
temp5 = temp4 + hiddenlayer2 * outputlayer

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = "progress_" + formatted_time + ".pkl"

make_path_dense(paths, 0.1)

step = 1 / 30
points_config = PointsConfig(5, 10, 50, 1, 5, 2, 5, 0.01, 0.1)

def fitness_func(ga_instance, solutions, solution_indices):

    # Create a neural network using the solution weights and biases
    with ThreadPool(pool_size) as p:
        networks = p.map(create_network, solutions)

    # Run the simulation and calculate the fitness score
    fitness = run_simulation(networks)

    # Return the fitness score
    return fitness


def run_simulation(networks):
    # Run the simulation with the given neural network
    # e.g. use a physics engine to simulate the drone movement
    # evaluate the fitness based on the problem definition

    simulation = Simulation(points_config, update_rays_every_n_frame=10)

    begin = time.time()

    timeout = 15.0

    simulation.clear()
    simulation.setup_world(random.choice(paths), len(networks), ignore_visualisation_objects=False)

    with ThreadPool(pool_size) as p:

        while simulation.timer <= timeout:

            inputs = [(networks[i], np.array([simulation.drones[i].ml_scaled])) for i in range(len(networks))]

            # print(inputs)

            predictions = p.map(lambda t: t[0](t[1], training=False), inputs)

            # print(predictions)

            for i in range(len(networks)):
                preds = predictions[i][0]
                print(preds)

                simulation.drones[i].propellers[0].force = max(-1.0, min(float(preds[0]), 1.0))
                simulation.drones[i].propellers[1].force = max(-1.0, min(float(preds[1]), 1.0))
                simulation.drones[i].propellers[2].force = max(-1.0, min(float(preds[2]), 1.0))
                simulation.drones[i].propellers[3].force = max(-1.0, min(float(preds[3]), 1.0))

            simulation.update(step)

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - begin

    points = [drone.points for drone in simulation.drones]

    print(len(networks), "networks", round(elapsed_time), "seconds", points)

    return points


def create_network(solution):
    # Create a neural network with 25 input nodes and 4 output nodes

    network = Sequential([
        Input(shape=(inputsize,)),
        Dense(hiddenlayer1, activation='relu', use_bias=True),
        Dense(hiddenlayer2, activation='relu', use_bias=True),
        Dense(outputlayer, activation='tanh', use_bias=True)
    ])

    # Set the weights and biases of the neural network to the values in the solution array
    start = 0
    # Get the shape of the layer's weights

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

num_generations = 100

mutation_percent_genes = 10

sol_per_pop = 30

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
                       num_genes=temp5 + outputlayer,
                       fitness_func=fitness_func,
                       fitness_batch_size=10,
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
                       # parallel_processing=["thread", 4]
                       )

ga_instance.run()

ga_instance.fitness_func = None

ga_instance.save(filename)