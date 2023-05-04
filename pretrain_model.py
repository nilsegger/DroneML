import time
import random

import keras
from pynput import keyboard
import pickle
import pychrono.core as chrono
import datetime

from train import *
from sim import Simulation, paths, make_path_dense, PointsConfig, SimulationVisual

model_save_file = 'pretrained_models/model_64_128_64_2023-05-04_12-42-01'

if model_save_file is None:
    hiddens = [64, 256, 64]
    original_model = create_model(17, hiddens, 4)
    train_model(original_model,
                    ['training/training.csv',
                     'training/training1.csv',
                     'training/training2.csv',
                     'training/training3.csv',
                     'training/training4.csv',
                     'training/training5.csv',
                     'training/training6.csv',
                     'training/training7.csv',
                     'training/training8.csv'
                     ], 100)

    now = datetime.datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
    hiddens_name = '_'.join([str(h) for h in hiddens])
    filename = "pretrained_models/model_" + hiddens_name + '_' + formatted

    original_model.save(filename)
else:
    original_model = keras.models.load_model(model_save_file)

model = original_model # get_model_from_solution(original_model, solution)

for i in range(100):
    y_point = random.randrange(0, 2000) / 100.0
    paths[0].append(chrono.ChVectorD(0.0, y_point, 0.0))
points_config = PointsConfig(100, 0.2, 10, 25, 100, 20, 10, 1, 5, 500, 100)
simulation = Simulation(points_config)
simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)

vis = SimulationVisual(simulation)

keys = {
    keyboard.Key.space: False,
    'r': False,
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



last = time.time()
while vis.is_window_open():
    end_time = time.time()
    elapsed_time = end_time - last
    last = time.time()

    predict_for_drone(model, simulation.drones[0])

    simulation.update(elapsed_time)
    vis.render()

    if keys['r']:

        paths[0].clear()

        for i in range(100):
            y_point = random.randrange(0, 2000) / 100.0
            paths[0].append(chrono.ChVectorD(0.0, y_point, 0.0))

        simulation.clear()
        simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)
        vis.bind_from_simulation()