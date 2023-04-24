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
import random
import time


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

    def __init__(self, abrupt_penalty, points_for_target, crash_penalty, standstill_penalty,
                 standstill_timeout, min_distance_for_penalty, max_distance_for_penalty, distance_penalty, floor_penalty):
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


class Drone:
    l, h, w = (0.3475, 0.1077, 0.283)  # dimensions of DJI drone
    mass = 0.895
    max_force = chrono.ChVectorD(0.0, 4.5, 0.0)

    forward = chrono.ChVectorD(1, 0, 0)
    right = chrono.ChVectorD(0, 0, 1)
    up = chrono.ChVectorD(0, 1, 0)

    rays = [
        forward,
        # chrono.ChVectorD(-1, 0, 0),
        # up,
        chrono.ChVectorD(0, -1, 0),
        # right,
        # chrono.ChVectorD(0, 0, -1),
        # (forward + right).GetNormalized(),
        # (forward - right).GetNormalized(),
        # (-forward + right).GetNormalized(),
        # (-forward - right).GetNormalized(),
        (forward + (right / 2.0) + (up / 2.0)).GetNormalized(),
        (forward + (right / 2.0) - (up / 2.0)).GetNormalized(),
        (forward - (right / 2.0) + (up / 2.0)).GetNormalized(),
        (forward - (right / 2.0) - (up / 2.0)).GetNormalized(),
        # (-forward + right + up).GetNormalized(),
        # (-forward - right + up).GetNormalized(),
        # (-forward + right - up).GetNormalized(),
        # (-forward - right - up).GetNormalized(),
    ]

    ray_length = 10
    ray_length_squared = 100
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

    def __init__(self, sys, origin_object, world_scale, path, points_config: PointsConfig):

        self.collision_system = sys.GetCollisionSystem()
        self.world_scale = world_scale

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
        self._rays_lengths = [0.0] * len(self.rays)

        self.ray_shapes = []

        if origin_object is not None:
            for i in range(len(self.rays)):
                mpathasset = chrono.ChLineShape()
                mpathasset.SetColor(self.color)
                origin_object.AddVisualShape(mpathasset)
                self.ray_shapes.append(mpathasset)

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

        # Calculate the parameter value 't' where the projection of point_vec onto line_vec is at its maximum
        t = max(min(point_vec.Dot(line_vec) / line_vec.Length2(), 1), 0)

        # Calculate the closest point on the line segment to the given point
        closest_point = line_start + line_vec * t

        direction = (line_end - line_start).GetNormalized()

        distance = (line_end - closest_point).Length()
        offset = 0.1

        if offset > distance:
            offset = distance

        self._target = closest_point + (direction * offset)

    @property
    def target(self):
        return self._target

    def create_ray_line(self, start, end):
        mpath = chrono.ChLinePath()
        mseg1 = chrono.ChLineSegment(start, end)
        mpath.AddSubLine(mseg1)
        mpath.Set_closed(False)
        return mpath

    def update_rays(self):

        start_point = self.body.GetPos()

        self._rays_lengths.clear()

        for i in range(len(self.rays)):
            direction = self.rays[i]
            end_point = self.body.GetPos() + self.body.TransformDirectionLocalToParent(direction) * self.ray_length

            ray_result = chrono.ChRayhitResult()
            self.collision_system.RayHit(start_point, end_point, ray_result)

            # Check if there was a collision
            if ray_result.hit:
                self._rays_lengths.append((ray_result.abs_hitPoint - start_point).Length2())

                if len(self.ray_shapes) > 0:
                    self.ray_shapes[i].SetLineGeometry(self.create_ray_line(start_point, ray_result.abs_hitPoint))
                    self.ray_shapes[i].SetColor(chrono.ChColor(1, 0, 0))
            else:
                self._rays_lengths.append(self.ray_length_squared)

                if len(self.ray_shapes) > 0:
                    self.ray_shapes[i].SetLineGeometry(self.create_ray_line(start_point, end_point))
                    self.ray_shapes[i].SetColor(self.color)

    def fitness_update(self, step):

        # Smoothness
        thresh = 3
        current_rotation_dt = self.body.GetRot_dt()

        rot_sum = abs(current_rotation_dt.e0) + abs(current_rotation_dt.e1) + abs(current_rotation_dt.e2) + abs(
            current_rotation_dt.e3)

        if rot_sum >= thresh:
            self.points -= self.points_config.abrupt_penalty

        # Target following
        target_thresh = 0.4

        if self.path_next < len(self.path):
            next_target = self.path[self.path_next]
            distance = (next_target - self.body.GetPos()).Length2()

            if distance < target_thresh * target_thresh:  # target was hit
                self.points += self.points_config.points_for_target
                self.path_next += 1

            if distance > self.points_config.min_distance_for_penalty:
                distance = min(self.points_config.max_distance_for_penalty, distance)
                self.points -= self.points_config.distance_penalty / self.points_config.max_distance_for_penalty * distance

        if not self.body.GetContactForce().IsNull() and self.path_next < len(self.path):
            self.points -= self.points_config.floor_penalty

        collision_force_thresh = 500 * 500

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
                self.body.GetPos().z, self.body.GetPos_dt().x, self.body.GetPos_dt().y, self.body.GetPos_dt().z, self.body.GetRot().e0, self.body.GetRot().e1, self.body.GetRot().e2,
                self.body.GetRot().e3, self.body.GetRot_dt().e0, self.body.GetRot_dt().e1, self.body.GetRot_dt().e2, self.body.GetRot_dt().e3] + self._rays_lengths

    @property
    def ml_scaled(self):
        return [
            self.target.x / self.world_scale, self.target.y / self.world_scale, self.target.z / self.world_scale,
            self.body.GetPos().x / self.world_scale, self.body.GetPos().y / self.world_scale, self.body.GetPos().z / self.world_scale,
            self.body.GetPos_dt().x / self.max_speed, self.body.GetPos_dt().y / self.max_speed, self.body.GetPos_dt().z / self.max_speed,
            self.body.GetRot().e0, self.body.GetRot().e1, self.body.GetRot().e2, self.body.GetRot().e3,
            self.body.GetRot_dt().e0, self.body.GetRot_dt().e1, self.body.GetRot_dt().e2, self.body.GetRot_dt().e3
            ] + [ray / self.ray_length_squared for ray in self._rays_lengths]



class Simulation:

    def __init__(self, points_config: PointsConfig, update_rays_every_n_frame=0):

        self.sys = chrono.ChSystemNSC()
        self.lines_parent = None
        self.drones = []
        self.points_config = points_config
        self.timer = 0.0
        self.update_rays_every_n_frame = update_rays_every_n_frame
        self.ray_update_counter = 0

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
            self.drones.append(Drone(self.sys, self.lines_parent, world_scale, path, self.points_config))

        self.update()

    def update(self, time_step=5e-3):

        for drone in self.drones:
            drone.update(time_step)

        self.timer += time_step
        self.sys.DoStepDynamics(time_step)

        self.ray_update_counter += 1

        if self.update_rays_every_n_frame == 0 or self.ray_update_counter == self.update_rays_every_n_frame:

            for drone in self.drones:
                drone.body.SetCollide(False)

            for drone in self.drones:
                drone.update_rays()

            for drone in self.drones:
                drone.body.SetCollide(True)

            self.ray_update_counter = 0

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
    elif keys['x']:
        for prop in drone.propellers:
            prop.force = 0.55
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


paths = [

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

]

if __name__ == '__main__':

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)

    listener.start()

    make_path_dense(paths, 0.1)

    simulation = Simulation(PointsConfig(5, 100, 10, 1, 10.0, 4, 10, 0.05, 0.1), update_rays_every_n_frame=6)
    simulation.setup_world(random.choice(paths), 1, ignore_visualisation_objects=False)

    window = SimulationVisual(simulation)

    last = time.time()

    frames = 0
    frames_time = 0

    last_points = 0

    current_drone = 0
    while window.is_window_open():

        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - last
        last = time.time()

        frames += 1
        frames_time += elapsed_time
        simulation.update(elapsed_time)

        drones_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        for i in range(len(drones_char)):
            if keys[drones_char[i]]:
                current_drone = i
        DroneManualInput(simulation.drones[current_drone])

        window.render()

        if last_points != simulation.drones[current_drone].points:
            last_points = simulation.drones[current_drone].points
            print(last_points)

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
