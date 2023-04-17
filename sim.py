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
import numpy as np
import random

# ---------------------------------------------------------------------
#
#  Create the simulation sys and add items
#

class Propeller:

    def __init__(self, position, force=0.0):
        self.position = position
        self.force = force


class DroneSimulation:
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

    def __init__(self, points, abrupt_penalty, points_for_target, crash_penalty, path_points):

        self.path = path_points[1:]
        # self.smooth_points = smooth_points
        self.abrupt_penalty = abrupt_penalty
        self.points = points
        self.points_for_target = points_for_target
        self.crash_penalty = crash_penalty
        self.timer = 0

        self.sys = chrono.ChSystemNSC()

        material = chrono.ChMaterialSurfaceNSC()
        material.SetFriction(0.3)
        material.SetCompliance(0)

        mfloor = chrono.ChBodyEasyBox(20, 0.2, 20, 1000)
        mfloor.SetBodyFixed(True)
        mfloor.SetCollide(True)
        mfloor.GetCollisionModel().AddBox(material, 10, 0.1, 10)
        mfloor.GetCollisionModel().BuildModel()
        self.sys.Add(mfloor)

        self.drone_x, self.drone_y, self.drone_z = (0.3475, 0.1077, 0.283)  # dimensions of DJI drone
        self.drone_kg = 0.895
        self.drone = chrono.ChBodyEasyBox(self.drone_x, self.drone_y, self.drone_z,
                                          self.drone_kg / self.drone_x / self.drone_y / self.drone_z)
        self.drone.SetMass(0.895)
        self.drone.SetPos(chrono.ChVectorD(0, self.drone_y + 0.05, 0))
        self.drone.GetCollisionModel().AddBox(material, self.drone_x / 2.0, self.drone_y / 2.0, self.drone_z / 2.0)
        self.drone.SetCollide(True)

        vis_drone = self.drone.GetVisualShape(0)
        vis_drone.SetColor(chrono.ChColor(1, 0, 0))

        self.sys.Add(self.drone)

        self.mray = chrono.ChBodyEasyBox(0, 0.0, 0, 0)
        self.mray.SetBodyFixed(True)
        self.sys.Add(self.mray)

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
            self.mray.AddVisualShape(mpathasset)
            self.drone_ray_shapes.append(mpathasset)

        drone_line_path = chrono.ChLinePath()

        for i in range(len(path_points) - 1):
            seg = chrono.ChLineSegment(path_points[i], path_points[i + 1])
            drone_line_path.AddSubLine(seg)

        drone_line_path.Set_closed(False)

        drone_path_shape = chrono.ChLineShape()
        drone_path_shape.SetLineGeometry(drone_line_path)
        drone_path_shape.SetColor(chrono.ChColor(0, 0, 1))

        mfloor.AddVisualShape(drone_path_shape)

        # ---------------------------------------------------------------------
        #
        #  Create an Irrlicht application to visualize the sys
        #

        self.vis = chronoirr.ChVisualSystemIrrlicht()
        self.vis.AttachSystem(self.sys)
        self.vis.SetWindowSize(1024, 768)
        self.vis.SetWindowTitle('DroneML')
        self.vis.Initialize()
        self.vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
        self.vis.AddSkyBox()
        camera_id = self.vis.AddCamera(chrono.ChVectorD(-20, 5, 0), chrono.ChVectorD(0, 2, 0))
        self.vis.AddTypicalLights()

        self.max_force = chrono.ChVectorD(0.0, 10.0, 0.0)  # Example force in the negative z-direction

        self.propellers = (
            Propeller(chrono.ChVectorD(self.drone_x / 2.0, 0, -self.drone_z / 2.0)),
            Propeller(chrono.ChVectorD(self.drone_x / 2.0, 0, self.drone_z / 2.0)),
            Propeller(chrono.ChVectorD(-self.drone_x / 2.0, 0, self.drone_z / 2.0)),
            Propeller(chrono.ChVectorD(-self.drone_x / 2.0, 0, -self.drone_z / 2.0))
        )

    def CreateLine(self, start, end):
        mpath = chrono.ChLinePath()
        mseg1 = chrono.ChLineSegment(start, end)
        mpath.AddSubLine(mseg1)
        mpath.Set_closed(False)
        return mpath

    def DroneSensors(self):
        start_point = self.drone.GetPos()

        results = []

        for i in range(len(self.drone_ray_dirs)):
            dir = self.drone_ray_dirs[i]
            ray_length = 10.0
            end_point = self.drone.GetPos() + self.drone.TransformDirectionLocalToParent(dir) * ray_length

            collision_system = self.sys.GetCollisionSystem()

            ray_result = chrono.ChRayhitResult()
            collision_system.RayHit(start_point, end_point, ray_result)

            # Check if there was a collision
            if ray_result.hit:
                # print(ray_result.abs_hitPoint)
                self.drone_ray_shapes[i].SetLineGeometry(self.CreateLine(start_point, ray_result.abs_hitPoint))
                self.drone_ray_shapes[i].SetColor(chrono.ChColor(1, 0, 0))
                results.append((ray_result.abs_hitPoint - start_point).Length2())
            else:
                self.drone_ray_shapes[i].SetLineGeometry(self.CreateLine(start_point, end_point))
                self.drone_ray_shapes[i].SetColor(chrono.ChColor(1, 1, 1))
                results.append(ray_length * ray_length)

        return results

    def DronePosition(self):
        return [self.drone.GetPos().x, self.drone.GetPos().y, self.drone.GetPos().z]

    def DroneRotation(self):
        return [self.drone.GetRot().e0, self.drone.GetRot().e1, self.drone.GetRot().e2, self.drone.GetRot().e3]

    def window_open(self):
        return self.vis.Run()

    def Render(self):
        self.vis.BeginScene()
        self.vis.Render()
        self.vis.EnableCollisionShapeDrawing(True)
        self.vis.EndScene()

    def Update(self, TimeStep=5e-3):

        yaw = (self.propellers[1].force + self.propellers[3].force) - (
                self.propellers[0].force + self.propellers[2].force)
        yaw_multiplier = 10.0

        current_rotation = self.drone.GetRot()

        # Define desired rotation angle (in radians)
        desired_rotation_angle = math.radians(yaw_multiplier * yaw)

        desired_rotation = chrono.ChQuaternionD()
        desired_rotation.Q_from_AngAxis(desired_rotation_angle,
                                        chrono.ChVectorD(0, 1,
                                                         0))  # Rotate by desired_rotation_angle radians about y-axis

        # Multiply current rotation by desired rotation to get new rotation
        new_rotation = current_rotation * desired_rotation

        self.drone.SetRot(new_rotation)

        self.drone.Empty_forces_accumulators()

        for prop in self.propellers:
            self.drone.Accumulate_force(self.max_force * prop.force, prop.position, True)

        self.sys.DoStepDynamics(TimeStep)

        self.timer += TimeStep

    def fitness_update(self):

        # Smoothness
        thresh = 3
        current_rotation_dt = self.drone.GetRot_dt()

        rot_sum = abs(current_rotation_dt.e0) + abs(current_rotation_dt.e1) + abs(current_rotation_dt.e2) + abs(
            current_rotation_dt.e3)

        if rot_sum >= thresh:
            self.points -= self.abrupt_penalty

        target_thresh = 1.0

        if len(self.path) > 0:
            next_target = self.path[0]
            distance = (next_target - self.drone.GetPos()).Length2()
            if distance < target_thresh:
                self.points += self.points_for_target
                self.path = self.path[1:]

        collision_force_thresh = 500 * 500

        if self.drone.GetContactForce().Length2() >= collision_force_thresh:
            self.points -= self.crash_penalty

    def fitness_final(self):
        # distance to target
        # speed
        # Crash
        pass

    def close(self):
        self.sys.Clear()
        # self.vis.GetDevice().closeDevice()
        # self.vis.GetDevice().drop()


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
    'e': False,
    't': False
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


def DroneManualInput(sim):
    hover_force_mult = 0.8  # Example force in the negative z-direction

    if keys[keyboard.Key.space]:
        # point = chrono.ChVectorD(drone_x / 2.0, 0, -drone_z / 2.0)  # Example force in the negative z-direction

        for prop in sim.propellers:
            prop.force = 1.0

    elif keys['w']:
        sim.propellers[0].force = hover_force_mult
        sim.propellers[1].force = hover_force_mult

        sim.propellers[2].force = 1.0
        sim.propellers[3].force = 1.0
    elif keys['s']:

        sim.propellers[2].force = hover_force_mult
        sim.propellers[3].force = hover_force_mult

        sim.propellers[0].force = 1.0
        sim.propellers[1].force = 1.0

    elif keys['d']:

        sim.propellers[1].force = hover_force_mult
        sim.propellers[2].force = hover_force_mult

        sim.propellers[0].force = 1.0
        sim.propellers[3].force = 1.0

    elif keys['a']:

        sim.propellers[0].force = hover_force_mult
        sim.propellers[3].force = hover_force_mult

        sim.propellers[1].force = 1.0
        sim.propellers[2].force = 1.0
    elif keys['q']:
        sim.propellers[1].force = 1.0
        sim.propellers[3].force = 1.0

        sim.propellers[0].force = hover_force_mult
        sim.propellers[2].force = hover_force_mult
    elif keys['e']:
        sim.propellers[0].force = 1.0
        sim.propellers[2].force = 1.0

        sim.propellers[1].force = hover_force_mult
        sim.propellers[3].force = hover_force_mult
    else:
        sim.propellers[0].force = 0.0
        sim.propellers[1].force = 0.0
        sim.propellers[2].force = 0.0
        sim.propellers[3].force = 0.0


paths = [

    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(0, 2.0, 0),
        chrono.ChVectorD(0, 4.0, 0.0),
        chrono.ChVectorD(0, 6.0, 0.0),
        chrono.ChVectorD(0, 8.0, 0.0),
        chrono.ChVectorD(0, 10.0, 0.0),
        chrono.ChVectorD(0.5, 12.0, 0.0),
        chrono.ChVectorD(2.0, 14.0, 0.0),
        chrono.ChVectorD(4.0, 16.0, 0.0),
        chrono.ChVectorD(6.0, 16.0, 0.0),
        chrono.ChVectorD(8.0, 16.0, 0.0),
    ],

    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(0, 2.0, 0),
        chrono.ChVectorD(1.0, 2.0, 0.0),
        chrono.ChVectorD(0, 4.0, 3.0),
        chrono.ChVectorD(3.0, 6.0, 1.0),
        chrono.ChVectorD(0, 8.0, 0.0),
        chrono.ChVectorD(0.5, 10.0, 0.0),
        chrono.ChVectorD(2.0, 12.0, 0.0),
        chrono.ChVectorD(4.0, 14.0, 0.0),
        chrono.ChVectorD(6.0, 16.0, 3.0),
        chrono.ChVectorD(8.0, 16.0, 4.0),
    ],

    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(2.0, 2.0, 0),
        chrono.ChVectorD(4.0, 4.0, 0.0),
        chrono.ChVectorD(6.0, 6.0, 0.0),
        chrono.ChVectorD(8.0, 8.0, 0.0),
        chrono.ChVectorD(10.0, 10.0, 0.0),
        chrono.ChVectorD(12.0, 12.0, 0.5),
        chrono.ChVectorD(10.0, 14.0, 2.0),
        chrono.ChVectorD(8.0, 16.0, 4.0),
        chrono.ChVectorD(6.0, 16.0, 6.0),
        chrono.ChVectorD(2.0, 16.0, 8.0),
    ],

    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(0, 2.0, 1),
        chrono.ChVectorD(0, 4.0, 4),
        chrono.ChVectorD(2, 6.0, 5),
        chrono.ChVectorD(0, 8.0, 2),
        chrono.ChVectorD(0, 10.0, 3),
        chrono.ChVectorD(4, 12.0, 7),
        chrono.ChVectorD(0.0, 14.0, 3),
        chrono.ChVectorD(0.0, 16.0, 9),
    ],

    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(0, 2.0, 0),
        chrono.ChVectorD(0, 4.0, 0.0),
        chrono.ChVectorD(0, 6.0, 0.0),
        chrono.ChVectorD(0, 8.0, 0.0),
        chrono.ChVectorD(0, 10.0, 0.0),
        chrono.ChVectorD(1, 10.0, 1),
        chrono.ChVectorD(2, 10.0, 2),
        chrono.ChVectorD(3, 10.0, 3),
        chrono.ChVectorD(4, 10.0, 4),
        chrono.ChVectorD(5, 10.0, 5),
        chrono.ChVectorD(6, 10.0, 6),
    ],

    [
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(1, 2.0, 0),
        chrono.ChVectorD(1, 4.0, 1.0),
        chrono.ChVectorD(1, 6.0, 2.0),
        chrono.ChVectorD(1, 8.0, 3.0),
        chrono.ChVectorD(1, 10.0, 3.0),
        chrono.ChVectorD(1, 12.0, 3.5),
        chrono.ChVectorD(1.0, 14.0, 3.0),
        chrono.ChVectorD(1.0, 16.0, 3.0),
    ],

]

if __name__ == '__main__':


    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)

    listener.start()


    while True:

        simulation = DroneSimulation(35, 0.1, 15, 100,
                                        random.choice(paths)
                                     )

        while simulation.window_open():
            simulation.Update()
            DroneManualInput(simulation)
            simulation.DroneSensors()
            simulation.Render()

            simulation.fitness_update()
            # print(simulation.drone.GetRotAxis().x, simulation.drone.GetRotAxis().y, simulation.drone.GetRotAxis().z)
            # print(simulation.drone.GetRot_dt())

            if keys['t']:
                break

        simulation.close()
