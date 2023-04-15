# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2019 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =============================================================================


import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from pynput import keyboard
import math

# ---------------------------------------------------------------------
#
#  Create the simulation sys and add items
#

sys = chrono.ChSystemNSC()

material = chrono.ChMaterialSurfaceNSC()
material.SetFriction(0.3)
material.SetCompliance(0)

mfloor = chrono.ChBodyEasyBox(3, 0.2, 3, 1000)
mfloor.SetBodyFixed(True)
mfloor.SetCollide(True)
mfloor.GetCollisionModel().AddBox(material, 1.5, 0.1, 1.5)
mfloor.GetCollisionModel().BuildModel()
sys.Add(mfloor)

drone_x, drone_y, drone_z = (0.3475, 0.1077, 0.283)  # dimensions of DJI drone
drone_kg = 0.895
drone = chrono.ChBodyEasyBox(drone_x, drone_y, drone_z, drone_kg / drone_x / drone_y / drone_z)
drone.SetMass(0.895)
drone.SetPos(chrono.ChVectorD(0, 1, 0))
drone.GetCollisionModel().AddBox(material, drone_x / 2.0, drone_y / 2.0, drone_z / 2.0)
drone.SetCollide(True)

vis_drone = drone.GetVisualShape(0)
vis_drone.SetColor(chrono.ChColor(1, 0, 0))

sys.Add(drone)

mray = chrono.ChBodyEasyBox(0, 0.0, 0, 0)
mray.SetBodyFixed(True)
sys.Add(mray)

drone_ray_shapes = []
forward = chrono.ChVectorD(1, 0, 0)
right = chrono.ChVectorD(0, 0, 1)
up = chrono.ChVectorD(0, 1, 0)

drone_ray_dirs = [
    forward,
    chrono.ChVectorD(-1, 0, 0),
    up,
    chrono.ChVectorD(0, -1, 0),
    right,
    chrono.ChVectorD(0, 0, -1),
    (forward + right).GetNormalized(),
    (forward - right).GetNormalized(),
    (-forward + right).GetNormalized(),
    (-forward - right).GetNormalized(),
    (forward + right + up).GetNormalized(),
    (forward + right - up).GetNormalized(),
    (forward - right + up).GetNormalized(),
    (forward - right - up).GetNormalized(),
    (-forward + right + up).GetNormalized(),
    (-forward - right + up).GetNormalized(),
    (-forward + right - up).GetNormalized(),
    (-forward - right - up).GetNormalized(),
]

for i in range(18):
    # mpath = chrono.ChLinePath()
    # mseg1 = chrono.ChLineSegment(chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(0, 0, 0))
    # mpath.AddSubLine(mseg1)
    # mpath.Set_closed(False)

    # Create a ChLineShape, a visualization asset for lines.
    # The ChLinePath is a special type of ChLine and it can be visualized.
    mpathasset = chrono.ChLineShape()
    # mpathasset.SetLineGeometry(mpath)
    mpathasset.SetColor(chrono.ChColor(1, 1, 1))
    mray.AddVisualShape(mpathasset)
    drone_ray_shapes.append(mpathasset)

# ---------------------------------------------------------------------
#
#  Create an Irrlicht application to visualize the sys
#

vis = chronoirr.ChVisualSystemIrrlicht()
vis.AttachSystem(sys)
vis.SetWindowSize(1024, 768)
vis.SetWindowTitle('Paths demo')
vis.Initialize()
vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
vis.AddSkyBox()
camera_id = vis.AddCamera(chrono.ChVectorD(-5, 2, 0), chrono.ChVectorD(0, 2, 0))
vis.AddTypicalLights()

# ---------------------------------------------------------------------
#
#  Run the simulation
#

keys = {
    keyboard.Key.space: False,
    'w': False,
    's': False,
    'a': False,
    'd': False,
    'q': False,
    'e': False
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


class Propeller:

    def __init__(self, position, force=0.0):
        self.position = position
        self.force = force


max_force = chrono.ChVectorD(0.0, 10.0, 0.0)  # Example force in the negative z-direction
local = True

propellers = (
    Propeller(chrono.ChVectorD(drone_x / 2.0, 0, -drone_z / 2.0)),
    Propeller(chrono.ChVectorD(drone_x / 2.0, 0, drone_z / 2.0)),
    Propeller(chrono.ChVectorD(-drone_x / 2.0, 0, drone_z / 2.0)),
    Propeller(chrono.ChVectorD(-drone_x / 2.0, 0, -drone_z / 2.0))
)


def DroneManualInput():
    hover_force_mult = 0.8  # Example force in the negative z-direction

    if keys[keyboard.Key.space]:
        # point = chrono.ChVectorD(drone_x / 2.0, 0, -drone_z / 2.0)  # Example force in the negative z-direction

        for prop in propellers:
            prop.force = 1.0

    elif keys['w']:
        propellers[0].force = hover_force_mult
        propellers[1].force = hover_force_mult

        propellers[2].force = 1.0
        propellers[3].force = 1.0
    elif keys['s']:

        propellers[2].force = hover_force_mult
        propellers[3].force = hover_force_mult

        propellers[0].force = 1.0
        propellers[1].force = 1.0

    elif keys['d']:

        propellers[1].force = hover_force_mult
        propellers[2].force = hover_force_mult

        propellers[0].force = 1.0
        propellers[3].force = 1.0

    elif keys['a']:

        propellers[0].force = hover_force_mult
        propellers[3].force = hover_force_mult

        propellers[1].force = 1.0
        propellers[2].force = 1.0
    elif keys['q']:
        propellers[1].force = 1.0
        propellers[3].force = 1.0

        propellers[0].force = hover_force_mult
        propellers[2].force = hover_force_mult
    elif keys['e']:
        propellers[0].force = 1.0
        propellers[2].force = 1.0

        propellers[1].force = hover_force_mult
        propellers[3].force = hover_force_mult
    else:
        propellers[0].force = 0.0
        propellers[1].force = 0.0
        propellers[2].force = 0.0
        propellers[3].force = 0.0


def CreateLine(start, end):
    mpath = chrono.ChLinePath()
    mseg1 = chrono.ChLineSegment(start, end)
    mpath.AddSubLine(mseg1)
    mpath.Set_closed(False)
    return mpath


def DroneSensors():
    start_point = drone.GetPos()

    for i in range(len(drone_ray_dirs)):
        dir = drone_ray_dirs[i]
        ray_length = 100.0
        end_point = drone.GetPos() + drone.TransformDirectionLocalToParent(dir) * ray_length

        collision_system = sys.GetCollisionSystem()

        ray_result = chrono.ChRayhitResult()
        collision_system.RayHit(start_point, end_point, ray_result)

        # Check if there was a collision
        if ray_result.hit:
            # print(ray_result.abs_hitPoint)
            drone_ray_shapes[i].SetLineGeometry(CreateLine(start_point, ray_result.abs_hitPoint))
            drone_ray_shapes[i].SetColor(chrono.ChColor(1, 0, 0))
        else:
            drone_ray_shapes[i].SetLineGeometry(CreateLine(start_point, end_point))
            drone_ray_shapes[i].SetColor(chrono.ChColor(1, 1, 1))


while vis.Run():

    vis.BeginScene()
    vis.Render()
    vis.EnableCollisionShapeDrawing(True)
    vis.EndScene()
    sys.DoStepDynamics(5e-3)

    DroneSensors()
    # vis.SetCameraTarget(drone.GetPos())
    DroneManualInput()

    yaw = (propellers[1].force + propellers[3].force) - (propellers[0].force + propellers[2].force)
    yaw_multiplier = 10.0

    current_rotation = drone.GetRot()

    # Define desired rotation angle (in radians)
    desired_rotation_angle = math.radians(yaw_multiplier * yaw)

    desired_rotation = chrono.ChQuaternionD()
    desired_rotation.Q_from_AngAxis(desired_rotation_angle,
                                    chrono.ChVectorD(0, 1, 0))  # Rotate by desired_rotation_angle radians about y-axis

    # Multiply current rotation by desired rotation to get new rotation
    new_rotation = current_rotation * desired_rotation

    drone.SetRot(new_rotation)

    drone.Empty_forces_accumulators()

    for prop in propellers:
        drone.Accumulate_force(max_force * prop.force, prop.position, local)
