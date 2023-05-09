import time
import random
from pynput import keyboard
import pychrono.core as chrono

from train import *
from sim import Simulation, PointsConfig, SimulationVisual

paths = [
    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(0, 5, 0),
        chrono.ChVectorD(0, 3, 0),
        chrono.ChVectorD(0, 5, 0),
        chrono.ChVectorD(0, 1, 0),
        chrono.ChVectorD(0, 7, 0)
    ]
]

instance_file = 'output/progress_gen_150_2023-05-07_19-17-15'  # no extension!

ga_instance = pygad.load(instance_file)

solutions = ga_instance.population

# make_path_dense(paths, 0.1)

points_config = PointsConfig(100, 0.2, 10, 25, 100, 20, 10, 1, 5, 500, 100)
simulation = Simulation(points_config)
simulation.setup_world(random.choice(paths), len(solutions), ignore_visualisation_objects=False)

vis = SimulationVisual(simulation)

original_model = create_model(17, [64, 128, 64], 4)

model_index = 0
models = [get_model_from_solution(original_model, solution) for solution in solutions]

keys = {
    keyboard.Key.space: False,
    'n': False,
}

keys_clicked = {
    'n': False,
    'a': False,
    'r': False
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

show_all = True

print("Press a to switch between show all and only show one")
print("Press n to view next model")
print("Press r to reset")

while vis.is_window_open():
    end_time = time.time()
    elapsed_time = end_time - last
    last = time.time()

    if not show_all:
        predict_for_drone(models[model_index], simulation.drones[0])
    else:
        for i in range(len(models)):
            predict_for_drone(models[i], simulation.drones[i])
    simulation.update(elapsed_time)
    vis.render()

    if not show_all and keys_clicked['n']:
        keys_clicked['n'] = False
        model_index = (model_index + 1) % len(solutions)
        simulation.clear()
        simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)
        vis.bind_from_simulation()
        print("Showing solution", model_index)

    if keys_clicked['a']:
        show_all = not show_all
        keys_clicked['a'] = False

        simulation.clear()
        simulation.setup_world(random.choice(paths), 1 if not show_all else len(models),
                               ignore_visualisation_objects=False)
        vis.bind_from_simulation()

    if keys_clicked['r']:
        keys_clicked['r'] = False
        simulation.clear()
        simulation.setup_world(random.choice(paths), 1 if not show_all else len(models),
                               ignore_visualisation_objects=False)
        vis.bind_from_simulation()