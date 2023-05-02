import time
import random
import pickle

from train import *
from sim import Simulation, paths, make_path_dense, PointsConfig, SimulationVisual

solution_file = 'output/best_solution_gen_100_2023-05-02_17-27-29.pkl'

with open(solution_file, "rb") as f:
    solution = pickle.load(f)

make_path_dense(paths, 0.1)

points_config = PointsConfig(100, 0.2, 10, 25, 100, 20, 10, 1, 5, 500, 100)
simulation = Simulation(points_config)
simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)

vis = SimulationVisual(simulation)

original_model = create_model(17, [7], 4)
model = get_model_from_solution(original_model, solution)

last = time.time()
while vis.is_window_open():
    end_time = time.time()
    elapsed_time = end_time - last
    last = time.time()

    predict_for_drone(model, simulation.drones[0])

    simulation.update(elapsed_time)
    vis.render()
