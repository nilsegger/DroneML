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
import numpy
import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from pynput import keyboard
import math
import random
import time
import numpy as np


def make_path_dense(path, mindisthreshold):
    for i in range(len(path)):
        numpoints = len(path[i])
        j = 0
        while j < numpoints - 1:
            if (path[i][j] - path[i][j + 1]).Length() > mindisthreshold:
                path[i].insert(j + 1, (path[i][j] + path[i][j + 1]) / 2)
                numpoints += 1
            else:
                j += 1
    return path


material = chrono.ChMaterialSurfaceNSC()
material.SetFriction(0.7)
material.SetCompliance(0)
material.SetRestitution(0.2)


class Propeller:

    def __init__(self, position, force=0.0):
        self.position = position
        self.force = force


class PointsConfig:

    def __init__(self, points_for_target, target_thresh_for_points, rotate_thresh_for_penalty, abrupt_penalty,
                 crash_penalty, standstill_penalty,
                 standstill_timeout, min_distance_for_penalty, max_distance_for_penalty, distance_penalty,
                 floor_penalty):
        self.abrupt_penalty = abrupt_penalty
        self.points_for_target = points_for_target
        self.crash_penalty = crash_penalty
        self.target_hit = points_for_target
        self.standstill_penalty = standstill_penalty
        self.standstill_timeout = standstill_timeout
        self.min_distance_for_penalty = min_distance_for_penalty
        self.max_distance_for_penalty = max_distance_for_penalty
        self.distance_penalty = distance_penalty
        self.floor_penalty = floor_penalty
        self.rotate_thresh = rotate_thresh_for_penalty
        self.target_thresh = target_thresh_for_points


class Drone:
    l, h, w = (0.3475, 0.1077, 0.283)  # dimensions of DJI drone
    mass = 0.895
    max_force = chrono.ChVectorD(0.0, 4.5, 0.0)

    max_speed = 100
    max_angular = 3.14159 * 10

    colors = [
        chrono.ChColor(1, 0, 0),  # Red
        chrono.ChColor(0, 1, 0),  # Green
        chrono.ChColor(0, 0, 1),  # Blue
        chrono.ChColor(1, 1, 0),  # Yellow
        chrono.ChColor(0, 1, 1),  # Cyan
        chrono.ChColor(1, 0, 1),  # Magenta
        chrono.ChColor(1, 0.5, 0),  # Orange
        chrono.ChColor(0.5, 1, 0),  # Lime
        chrono.ChColor(1, 0.75, 0.8),  # Pink
        chrono.ChColor(0.25, 0.88, 0.82)  # Turquoise
    ]

    def __init__(self, sys, origin_object, path, points_config: PointsConfig):

        self.collision_system = sys.GetCollisionSystem()

        self.points = 0
        self.standstill_timer = 0
        self.last_target_hit = 0
        self.points_config = points_config

        self.color = random.choice(self.colors)

        self.body = chrono.ChBodyEasyBox(self.l, self.h, self.w,
                                         self.mass / self.l / self.h / self.w)

        self.body.SetMass(0.895)
        self.body.SetPos(chrono.ChVectorD(0, self.h + 0.05, 0))
        self.body.GetCollisionModel().AddBox(material, self.l / 2.0, self.h / 2.0, self.w / 2.0)
        self.body.GetCollisionModel().SetFamily(2)
        self.body.GetCollisionModel().SetFamilyMaskNoCollisionWithFamily(2)
        self.body.SetCollide(True)

        self.body.SetLimitSpeed(True)
        self.body.SetMaxSpeed(self.max_speed)
        self.body.SetMaxWvel(self.max_angular)

        self.body.GetVisualShape(0).SetColor(self.color)

        sys.Add(self.body)

        self.propellers = (
            Propeller(chrono.ChVectorD(self.l / 2.0, self.h, -self.w / 2.0)),
            Propeller(chrono.ChVectorD(self.l / 2.0, self.h, self.w / 2.0)),
            Propeller(chrono.ChVectorD(-self.l / 2.0, self.h, self.w / 2.0)),
            Propeller(chrono.ChVectorD(-self.l / 2.0, self.h, -self.w / 2.0))
        )

        self.closest_point_on_path_visual = None
        if origin_object is not None:
            self.closest_point_on_path_visual = chrono.ChBodyEasySphere(0.2, 100)
            self.closest_point_on_path_visual.SetPos(chrono.ChVectorD(0, 0, 0))
            self.closest_point_on_path_visual.SetBodyFixed(True)
            self.closest_point_on_path_visual.GetVisualShape(0).SetColor(self.color)
            sys.Add(self.closest_point_on_path_visual)

        self.path = path
        self.path_next = 1

        self._target = chrono.ChVectorD(0, 0, 0)

    def update(self, time_step):
        yaw = (self.propellers[1].force + self.propellers[3].force) - (
                self.propellers[0].force + self.propellers[2].force)
        yaw_multiplier = 1.0

        current_rotation = self.body.GetRot()

        # Define desired rotation angle (in radians)
        desired_rotation_angle = math.radians(yaw_multiplier * yaw)

        desired_rotation = chrono.ChQuaternionD()
        desired_rotation.Q_from_AngAxis(desired_rotation_angle,
                                        chrono.ChVectorD(0, 1,
                                                         0))  # Rotate by desired_rotation_angle radians about y-axis

        # Multiply current rotation by desired rotation to get new rotation
        new_rotation = current_rotation * desired_rotation

        self.body.SetRot(new_rotation)

        self.body.Empty_forces_accumulators()

        for prop in self.propellers:
            self.body.Accumulate_force(self.max_force * prop.force, prop.position, True)

        self.target_update()

        if self.closest_point_on_path_visual is not None:
            self.closest_point_on_path_visual.SetPos(self.target)

        self.fitness_update(time_step)

    def target_update(self):

        if self.path_next >= len(self.path):
            self._target = chrono.ChVectorD(0, 0, 0)
            return

        line_start = self.path[self.path_next - 1]
        line_end = self.path[self.path_next]

        # Calculate the vector from line start to point
        line_vec = line_end - line_start
        point_vec = self.body.GetPos() - line_start

        if line_vec.Length() != 0:
            # Calculate the parameter value 't' where the projection of point_vec onto line_vec is at its maximum
            t = max(min(point_vec.Dot(line_vec) / line_vec.Length2(), 1), 0)
            # Calculate the closest point on the line segment to the given point
            closest_point = line_start + line_vec * t
        else:
            closest_point = line_start

        direction = (line_end - line_start).GetNormalized()

        distance = (line_end - closest_point).Length()
        offset = 1

        if offset > distance:
            offset = distance

        self._target = closest_point + (direction * offset)

    @property
    def target(self):
        return self._target

    def fitness_update(self, step):

        # Smoothness
        current_rotation_dt = self.body.GetRot_dt()

        rot_sum = abs(current_rotation_dt.e0) + abs(current_rotation_dt.e1) + abs(current_rotation_dt.e2) + abs(
            current_rotation_dt.e3)

        if rot_sum >= self.points_config.rotate_thresh * step:
            self.points -= self.points_config.abrupt_penalty * step

        if self.path_next < len(self.path):
            next_target = self.path[self.path_next]
            distance = (next_target - self.body.GetPos()).Length()

            if distance < self.points_config.target_thresh:  # target was hit
                self.points += self.points_config.points_for_target
                self.path_next += 1

            if distance > self.points_config.min_distance_for_penalty:
                self.points -= (self.points_config.distance_penalty * step) / (
                        self.points_config.max_distance_for_penalty - self.points_config.min_distance_for_penalty) * (
                                       distance - self.points_config.min_distance_for_penalty)

        if not self.body.GetContactForce().IsNull() and self.path_next < len(self.path):
            self.points -= self.points_config.floor_penalty * step

        collision_force_thresh = 250 * 250

        if self.body.GetContactForce().Length2() >= collision_force_thresh:
            self.points -= self.points_config.crash_penalty

        self.standstill_timer += step
        if self.standstill_timer > self.points_config.standstill_timeout and self.last_target_hit == self.path_next:
            self.standstill_timer = 0
            self.points -= self.points_config.standstill_penalty
        elif self.last_target_hit != self.path_next:
            self.standstill_timer = 0
            self.last_target_hit = self.path_next

    @property
    def ml(self):
        return [self.target.x, self.target.y, self.target.z, self.body.GetPos().x, self.body.GetPos().y,
                self.body.GetPos().z, self.body.GetPos_dt().x, self.body.GetPos_dt().y, self.body.GetPos_dt().z,
                self.body.GetRot().e0, self.body.GetRot().e1, self.body.GetRot().e2,
                self.body.GetRot().e3, self.body.GetRot_dt().e0, self.body.GetRot_dt().e1, self.body.GetRot_dt().e2,
                self.body.GetRot_dt().e3]


class Simulation:

    def __init__(self, points_config: PointsConfig):

        self.sys = chrono.ChSystemNSC()
        self.lines_parent = None
        self.drones = []
        self.points_config = points_config
        self.timer = 0.0

    def setup_world(self, path, n_drones, ignore_visualisation_objects=True):

        self.timer = 0.0

        world_scale = 50
        mfloor = chrono.ChBodyEasyBox(world_scale, 10, world_scale, 1000)
        mfloor.SetBodyFixed(True)
        mfloor.SetPos(chrono.ChVectorD(0, -5, 0))
        mfloor.SetCollide(True)
        mfloor.GetCollisionModel().AddBox(material, world_scale / 2, 5, world_scale / 2)
        mfloor.GetCollisionModel().BuildModel()
        self.sys.Add(mfloor)

        self.lines_parent = None

        if not ignore_visualisation_objects:
            self.lines_parent = chrono.ChBodyEasyBox(0, 0, 0, 0)
            self.lines_parent.SetBodyFixed(True)

            self.sys.Add(self.lines_parent)

            path_line = chrono.ChLinePath()

            for i in range(len(path) - 1):
                seg = chrono.ChLineSegment(path[i], path[i + 1])
                path_line.AddSubLine(seg)

            path_line.Set_closed(False)

            path_shape = chrono.ChLineShape()
            path_shape.SetLineGeometry(path_line)
            path_shape.SetColor(chrono.ChColor(0, 0, 1))

            self.lines_parent.AddVisualShape(path_shape)

        for i in range(n_drones):
            self.drones.append(Drone(self.sys, self.lines_parent, path, self.points_config))

        self.update()

    def update(self, time_step=5e-3):

        for drone in self.drones:
            drone.update(time_step)

        self.timer += time_step
        self.sys.DoStepDynamics(time_step)

    def clear(self):
        self.sys.Clear()
        self.drones.clear()


class SimulationVisual:

    def __init__(self, simulation):
        # ---------------------------------------------------------------------
        #
        #  Create an Irrlicht application to visualize the sys
        #

        self.vis = chronoirr.ChVisualSystemIrrlicht()
        self.vis.AttachSystem(simulation.sys)
        self.vis.SetWindowSize(1024, 768)
        self.vis.SetWindowTitle('DroneML')
        self.vis.Initialize()
        self.vis.AddLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
        self.vis.AddSkyBox()
        self.vis.AddCamera(chrono.ChVectorD(-20, 5, 0), chrono.ChVectorD(0, 2, 0))
        self.vis.AddTypicalLights()

    def bind_from_simulation(self):
        self.vis.BindAll()

    def is_window_open(self):
        return self.vis.Run()

    def render(self):
        self.vis.BeginScene()
        self.vis.Render()
        self.vis.EnableCollisionShapeDrawing(True)
        self.vis.EndScene()


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
    'x': False,
    'e': False,
    'r': False,
    'p': False,
    '0': False,
    '1': False,
    '2': False,
    '3': False,
    '4': False,
    '5': False,
    '6': False,
    '7': False,
    '8': False,
    '9': False,
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


def DroneManualInput(drone):
    hover_force_mult = 0.29  # Example force in the negative z-direction
    mov_force_mult = 0.3  # Example force in the negative z-direction

    if keys[keyboard.Key.space]:
        # point = chrono.ChVectorD(drone_x / 2.0, 0, -drone_z / 2.0)  # Example force in the negative z-direction
        for prop in drone.propellers:
            prop.force = 1.0
    elif keys['w']:
        drone.propellers[0].force = hover_force_mult
        drone.propellers[1].force = hover_force_mult

        drone.propellers[2].force = mov_force_mult
        drone.propellers[3].force = mov_force_mult
    elif keys['s']:

        drone.propellers[2].force = hover_force_mult
        drone.propellers[3].force = hover_force_mult

        drone.propellers[0].force = mov_force_mult
        drone.propellers[1].force = mov_force_mult

    elif keys['d']:

        drone.propellers[1].force = hover_force_mult
        drone.propellers[2].force = hover_force_mult

        drone.propellers[0].force = mov_force_mult
        drone.propellers[3].force = mov_force_mult

    elif keys['a']:
        drone.propellers[0].force = hover_force_mult
        drone.propellers[3].force = hover_force_mult

        drone.propellers[1].force = mov_force_mult
        drone.propellers[2].force = mov_force_mult
    elif keys['q']:
        drone.propellers[1].force = 1.0
        drone.propellers[3].force = 1.0

        drone.propellers[0].force = hover_force_mult
        drone.propellers[2].force = hover_force_mult
    elif keys['e']:
        drone.propellers[0].force = 1.0
        drone.propellers[2].force = 1.0

        drone.propellers[1].force = hover_force_mult
        drone.propellers[3].force = hover_force_mult
    else:
        drone.propellers[0].force = 0.0
        drone.propellers[1].force = 0.0
        drone.propellers[2].force = 0.0
        drone.propellers[3].force = 0.0


def StabiliseDrone(drone):
    rot = drone.body.GetRot().Q_to_Euler123()

    pitch = rot.z
    yaw = rot.y
    roll = rot.x

    # stabilise y
    vel_force = drone.body.GetPos_dt().y * drone.mass
    weight_force = drone.mass * 9.81 - vel_force
    for propeller in drone.propellers:
        propeller.force = min(1.0, max(0.0,
                                       weight_force / drone.max_force.y / 4.0))  # scale to 0 to 1 and /4 because of 4 props


def follow_y_path(drone):
    distance = drone.target.y - drone.body.GetPos().y

    time_it_takes_to_reach = math.sqrt((2.0 * math.fabs(distance)) / 9.81)
    required_acceleration = 2 * distance / (math.pow(time_it_takes_to_reach, 2))

    force_required = drone.mass * required_acceleration

    vel_force = drone.body.GetPos_dt().y * drone.mass
    weight_force = drone.mass * 9.81 - vel_force
    current_force = -weight_force

    if distance >= 0.0:
        # drone must fly up
        if current_force < force_required:

            force_wanted = force_required - current_force

            for propeller in drone.propellers:
                propeller.force = max(-1.0, min(1.0, 1.0 / drone.max_force.y / 4.0 * force_wanted))
        else:
            for propeller in drone.propellers:
                propeller.force = 0
    else:
        # drone must "fall"

        vel_force = drone.body.GetPos_dt().y * drone.mass
        weight_force = -vel_force

        if drone.body.GetPos_dt().y < 0.0 and math.fabs(weight_force) > drone.max_force.y:
            for propeller in drone.propellers:
                propeller.force = 1.0
        else:
            for propeller in drone.propellers:
                propeller.force = 0

paths = [
    [
    ]
]

"""

[
    chrono.ChVectorD(0, 0, 0),
    chrono.ChVectorD(0, 5, 0),
    chrono.ChVectorD(0, 3, 0),
    chrono.ChVectorD(0, 5, 0),
    chrono.ChVectorD(0, 0, 0)
]

[
        chrono.ChVectorD(0, 0, 0),
        chrono.ChVectorD(2, 2, 0),
        chrono.ChVectorD(2, 4, 2),
        chrono.ChVectorD(-4, 6, 2),
        chrono.ChVectorD(-4, 6, 5),
        chrono.ChVectorD(-4, 8, 5),
        chrono.ChVectorD(-4, 12, 5),
        chrono.ChVectorD(0, 12, 0),
        chrono.ChVectorD(0, 0, 0),
    ],
"""

if __name__ == '__main__':

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)

    listener.start()

    x_offset = random.randrange(0, 5000) / 100.0
    z_offset = random.randrange(0, 5000) / 100.0

    for i in range(1000):
        y_point = random.randrange(0, 5000) / 100.0
        paths[0].append(chrono.ChVectorD(x_offset, y_point, z_offset))

    # make_path_dense(paths, 1)

    points_config = PointsConfig(100, 0.2, 10, 25, 100, 20, 10, 1, 5, 500, 100)
    simulation = Simulation(points_config)
    simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)
    simulation.drones[0].body.SetPos(chrono.ChVectorD(x_offset, 0.02, z_offset))

    window = SimulationVisual(simulation)

    last = time.time()

    frames = 0
    frames_time = 0

    last_points = 0
    last_path_next = 1

    current_drone = 0

    training_data = []

    while window.is_window_open():

        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - last
        last = time.time()

        frames += 1
        frames_time += elapsed_time
        simulation.update(1/30)

        drones_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        for i in range(len(drones_char)):
            if keys[drones_char[i]]:
                current_drone = i

        follow_y_path(simulation.drones[current_drone])

        """
        if keys['x']:
            StabiliseDrone(simulation.drones[current_drone], elapsed_time)
        else:
            DroneManualInput(simulation.drones[current_drone])
        """

        if not simulation.drones[current_drone].target.IsNull():
            training_data.append(simulation.drones[current_drone].ml + [propeller.force for propeller in
                                                                    simulation.drones[current_drone].propellers])

        #if simulation.drones[current_drone].body.GetPos().x != x_offset or simulation.drones[current_drone].body.GetPos().z != z_offset:
            #print("Drone got offset.")
            #break

        if simulation.drones[current_drone].target.IsNull():
            print("Finished all points")
            break

        if keys['p']:
            window.render()

        if last_points != simulation.drones[current_drone].points:
            last_points = simulation.drones[current_drone].points
            # print(last_points)

        if last_path_next != simulation.drones[current_drone].path_next:
            last_path_next = simulation.drones[current_drone].path_next
            print(last_path_next)

        if frames == 100:
            timeout = 30.0
            time_step = 1 / 30
            avg_time_for_one_frame = frames_time / 100.0
            # print(timeout, "seconds of sim would take", timeout / time_step * avg_time_for_one_frame, "avg time for one frame", avg_time_for_one_frame)
            frames = 0
            frames_time = 0

        if keys['r']:
            simulation.clear()

            simulation.setup_world(random.choice(paths), 1, False)

            window.bind_from_simulation()

            frames = 0
            frames_time = 0

    np.savetxt("training.csv", numpy.asarray(training_data), delimiter=',')
