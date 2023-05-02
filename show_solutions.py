import time
import random
from pynput import keyboard

from train import *
from sim import Simulation, paths, make_path_dense, PointsConfig, SimulationVisual

instance_file = 'output/progress_gen_100_2023-05-02_17-27-23'  # no extension!

ga_instance = pygad.load(instance_file)

solutions = ga_instance.population

make_path_dense(paths, 0.1)

points_config = PointsConfig(100, 0.2, 10, 25, 100, 20, 10, 1, 5, 500, 100)
simulation = Simulation(points_config)
simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)

vis = SimulationVisual(simulation)

original_model = create_model(17, [7], 4)
model_index = 0
model = get_model_from_solution(original_model, solutions[model_index])

keys = {
    keyboard.Key.space: False,
    'n': False,
}

keys_clicked = {
    'n': False
}


def on_press(key):
    global keys

    try:
        keys[key.char] = True
        keys_clicked[key.char] = False
    except AttributeError:
        keys[key] = True
        keys_clicked[key] = False


def on_release(key):
    global keys
    try:
        keys[key.char] = False
        keys_clicked[key.char] = True
    except AttributeError:
        keys[key] = False
        keys_clicked[key] = True


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)

listener.start()

last = time.time()
while vis.is_window_open():
    end_time = time.time()
    elapsed_time = end_time - last
    last = time.time()

    predict_for_drone(model, simulation.drones[0])

    simulation.update(elapsed_time)
    vis.render()

    if keys_clicked['n']:
        keys_clicked['n'] = False
        model_index = (model_index + 1) % len(solutions)
        model = get_model_from_solution(original_model, solutions[model_index])
        simulation.clear()
        simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)
        vis.bind_from_simulation()
        print("Showing solution", model_index)
