import time
import random
import pickle

from train import *
from sim import Simulation, paths, make_path_dense, PointsConfig, SimulationVisual

solution_file = 'best_solution_gen_20_2023-04-24_19-40-37.pkl'

with open(solution_file, "rb") as f:
    solution = pickle.load(f)

make_path_dense(paths, 0.1)

simulation = Simulation(PointsConfig(1, 1, 1, 1, 1, 1, 1, 1, 1), 0)
simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)

vis = SimulationVisual(simulation)

original_model = create_model(23, 256, 256, 4)
model = get_model_from_solution(original_model, solution)

last = time.time()
while vis.is_window_open():
    end_time = time.time()
    elapsed_time = end_time - last
    last = time.time()

    predict_for_drone(model, simulation.drones[0])

    simulation.update(elapsed_time)
    vis.render()
