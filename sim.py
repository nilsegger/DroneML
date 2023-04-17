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
import time


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

    def SetupIrrlicht(self):
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

    def SetupWorld(self, path):

        self.path = path[1:]

        material = chrono.ChMaterialSurfaceNSC()
        material.SetFriction(0.7)
        material.SetCompliance(0)
        material.SetRestitution(0.2)

        mfloor = chrono.ChBodyEasyBox(50, 0.2, 50, 1000)
        mfloor.SetBodyFixed(True)
        mfloor.SetCollide(True)
        mfloor.GetCollisionModel().AddBox(material, 25, 0.1, 25)
        mfloor.GetCollisionModel().BuildModel()
        self.sys.Add(mfloor)

        self.drone = chrono.ChBodyEasyBox(self.drone_x, self.drone_y, self.drone_z,
                                          self.drone_kg / self.drone_x / self.drone_y / self.drone_z)

        self.drone.SetMass(0.895)
        self.drone.SetPos(chrono.ChVectorD(0, self.drone_y + 0.05, 0))
        self.drone.GetCollisionModel().AddBox(material, self.drone_x / 2.0, self.drone_y / 2.0, self.drone_z / 2.0)
        self.drone.SetCollide(True)

        vis_drone = self.drone.GetVisualShape(0)
        vis_drone.SetColor(chrono.ChColor(1, 0, 0))

        self.sys.Add(self.drone)

        mray = chrono.ChBodyEasyBox(0, 0.0, 0, 0)
        mray.SetBodyFixed(True)
        self.sys.Add(mray)

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
            self.drone_ray_shapes.append(mpathasset)

        drone_line_path = chrono.ChLinePath()

        for i in range(len(path) - 1):
            seg = chrono.ChLineSegment(path[i], path[i + 1])
            drone_line_path.AddSubLine(seg)

        drone_line_path.Set_closed(False)

        drone_path_shape = chrono.ChLineShape()
        drone_path_shape.SetLineGeometry(drone_line_path)
        drone_path_shape.SetColor(chrono.ChColor(0, 0, 1))

        mfloor.AddVisualShape(drone_path_shape)

        if self.vis is not None:
            self.vis.BindAll()

    def SetupPoints(self, points, abrupt_penalty, points_for_target, crash_penalty, standstill_penalty,
                    standstill_timeout, min_distance_for_penalty, max_distance_for_penalty, distance_penalty):
        self.abrupt_penalty = abrupt_penalty
        self.points = points
        self.points_for_target = points_for_target
        self.crash_penalty = crash_penalty
        self.timer = 0
        self.target_hit = 0
        self.standstill_penalty = standstill_penalty
        self.standstill_timeout = standstill_timeout
        self.standstill_timer = 0.0
        self.last_target_hit_counter = 0
        self.min_distance_for_penalty = min_distance_for_penalty * min_distance_for_penalty
        self.max_distance_for_penalty = max_distance_for_penalty * max_distance_for_penalty
        self.distance_penalty = distance_penalty

    def __init__(self):

        self.sys = chrono.ChSystemNSC()
        self.vis = None

        self.path = None
        self.abrupt_penalty = None
        self.points = None
        self.points_for_target = None
        self.crash_penalty = None
        self.timer = 0
        self.target_hit = 0
        self.standstill_penalty = None
        self.standstill_timeout = None

        self.max_force = chrono.ChVectorD(0.0, 4.5, 0.0)  # Example force in the negative z-direction

        self.drone_x, self.drone_y, self.drone_z = (0.3475, 0.1077, 0.283)  # dimensions of DJI drone
        self.drone_kg = 0.895
        self.drone = None

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

    def NextTarget(self):
        if len(self.path) > 0:
            return [self.path[0].x, self.path[0].y, self.path[0].z]

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
        yaw_multiplier = 1.0

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

    def fitness_update(self, step):

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
                self.target_hit += 1

            if distance > self.min_distance_for_penalty:
                print("Distance penalty!", self.distance_penalty / self.max_distance_for_penalty * distance)
                distance = min(self.max_distance_for_penalty, distance)
                self.points -= self.distance_penalty / self.max_distance_for_penalty * distance

        collision_force_thresh = 500 * 500

        if self.drone.GetContactForce().Length2() >= collision_force_thresh:
            self.points -= self.crash_penalty

        self.standstill_timer += step
        if self.standstill_timer > self.standstill_timeout and self.last_target_hit_counter == self.target_hit:
            self.standstill_timer = 0
            self.points -= self.standstill_penalty
        elif self.last_target_hit_counter != self.target_hit:
            self.standstill_timer = 0
            self.last_target_hit_counter = self.target_hit

    def fitness_final(self):
        # distance to target
        # speed
        # Crash
        pass

    def clear(self):
        self.sys.Clear()

        self.drone_ray_shapes.clear()


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
    hover_force_mult = 0.95  # Example force in the negative z-direction

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
        chrono.ChVectorD(0.5, 5.0, 0.5),
        chrono.ChVectorD(0.5, 6.0, 0.5),
        chrono.ChVectorD(1, 8.0, 1),
        chrono.ChVectorD(2, 8.0, 1),
        chrono.ChVectorD(3, 8.0, 2),
        chrono.ChVectorD(5, 9.0, 2),
        chrono.ChVectorD(7, 9.0, -2),
        chrono.ChVectorD(7, 9.0, 0),
        chrono.ChVectorD(10, 10.0, 0),
        chrono.ChVectorD(10, 9.0, 5),
        chrono.ChVectorD(10, 6.0, 5),
        chrono.ChVectorD(10, 3.0, 5),
        chrono.ChVectorD(10, 0.0, 8),
    ]
]

if __name__ == '__main__':

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)

    listener.start()

    simulation = DroneSimulation()

    simulation.SetupIrrlicht()

    simulation.SetupWorld(random.choice(paths))

    simulation.SetupPoints(35, 0.1, 15, 100, 10, 5.0, 2, 10, 1)

    last = time.time()

    last_points = 0

    while simulation.window_open():

        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - last
        last = time.time()

        simulation.Update(elapsed_time)

        DroneManualInput(simulation)
        simulation.DroneSensors()
        simulation.Render()

        simulation.fitness_update(elapsed_time)

        if last_points != simulation.points:
            print(simulation.points)
            last_points = simulation.points

        if keys['t']:
            simulation.clear()

            simulation.SetupWorld(random.choice(paths))

            simulation.SetupPoints(35, 0.1, 15, 100, 10, 5.0, 2, 10, 1)
