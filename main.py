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

print("Example: create a sys and visualize it in realtime 3D");

# The path to the Chrono data directory containing various assets (meshes, textures, data files)
# is automatically set, relative to the default location of this demo.
# If running from a different directory, you must change the path to the data directory with:
# chrono.SetChronoDataPath('path/to/data')

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

drone = chrono.ChBodyEasyBox(1, 0.2, 1, 1000)
drone.SetPos(chrono.ChVectorD(0, 1, 0))
drone.GetCollisionModel().AddBox(material, 0.5, 0.1, 0.5)
drone.SetCollide(True)

vis_drone_shape = chrono.ChBoxShape(chrono.ChBox(1, 0.2, 1))
vis_drone_shape.SetColor(chrono.ChColor(1, 0, 0))

vis_drone = chrono.ChVisualModel()
vis_drone.AddShape(vis_drone_shape)

drone.AddVisualShape(vis_drone_shape)

sys.Add(drone)

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
vis.AddCamera(chrono.ChVectorD(1, 4, 5), chrono.ChVectorD(0, 2, 0))
vis.AddTypicalLights()

# ---------------------------------------------------------------------
#
#  Run the simulation
#

while vis.Run():
    vis.BeginScene()
    vis.Render()
    vis.EnableCollisionShapeDrawing(True)
    vis.EndScene()
    sys.DoStepDynamics(5e-3)
