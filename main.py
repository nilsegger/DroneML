from sim import DroneSimulation

import numpy as np
import pygad
from keras import Sequential, Input
from keras.layers import Dense

# Define the problem
input_size = 25  # 3D coordinates of the drone
output_size = 4  # Output for the 4 thrusters
target = np.array([50, 50, 8])  # Destination coordinates
obstacles = []  # List of 3D coordinates of obstacles


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

    simulation = DroneSimulation()

    while simulation.window_open():

        rays = np.array(simulation.DroneSensors())
        rotation = np.array(simulation.DroneRotation())

        predictions = network.predict(rays + rotation + target)

        print(predictions)
        for i in range(4):
            simulation.propellers[i].force = predictions[i]

        simulation.Update()

        simulation.Render()

    score = 1.0
    return score


def create_network(solution):
    print(len(solution))
    print(solution)

    # Create a neural network with 25 input nodes and 4 output nodes
    network = Sequential([
        Input(shape=(25,)),
        Dense(12, activation='sigmoid', use_bias=True),
        Dense(4, activation='sigmoid', use_bias=True)
    ])

    # Set the weights and biases of the neural network to the values in the solution array
    start = 0
    for i, layer in enumerate(network.layers):
        # Get the shape of the layer's weights
        shape = layer.get_weights()[0].shape

        # Calculate the number of weights and biases for this layer
        num_weights = shape[0] * shape[1]
        num_biases = shape[1]
        print(num_weights)
        print(num_biases)
        # Get the weights and biases from the solution array
        end = start + num_weights
        weights = solution[start:end].reshape(shape)
        start = end
        end = start + num_biases
        biases = np.array(solution[start:end])
        start = end

        # Set the layer's weights and biases to the values from the solution array

        layer.set_weights([weights, biases])

    return network


# Create an initial population
num_solutions = 10
sol_per_pop = 5
"""
initial_population = np.random.uniform(low=-1, high=1,
                                       size=(num_solutions, 25 * 10)) # input_size * output_size
"""
# Create a genetic algorithm object and run it
ga_instance = pygad.GA(num_generations=2,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=25 * 12 + 12)

ga_instance.run()

# Retrieve the best solution
best_solution, best_solution_fitness = ga_instance.best_solution()
best_network = create_network(best_solution)
