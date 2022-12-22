from particle_system import ParticleSystem
from pbf_solver import PBF_Solver
from pbd_solver import PBD_Solver
import yaml
import taichi as ti
import numpy as np
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--cfg",type=str,default="pbd_config_2f2s.yml")
args=parser.parse_args()

ti.init(
    arch=ti.gpu
)

with open(args.cfg, "r") as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

particle_grid = ParticleSystem(config)
solver = PBD_Solver(particle_grid, config)

window = ti.ui.Window(f'PBF3D ({particle_grid.total_particle_num} particles)', res = (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()

camera = ti.ui.Camera()
camera.position(*(particle_grid.domain_sz * 1.5))
camera.up(0.0, 1.0, 0.0)
camera.lookat(0.0, 0.0, 0.0)
camera.fov(70)
scene.set_camera(camera)

frame_count = 0
gui = window.get_gui()

animating = False
while window.running:
    ad = 0.0
    ws = 0.0
    window.get_event(ti.ui.PRESS)
    if window.is_pressed(ti.ui.LEFT): ad = -1
    elif window.is_pressed(ti.ui.RIGHT): ad = 1

    if window.is_pressed(ti.ui.UP): ws = 1
    elif window.is_pressed(ti.ui.DOWN): ws = -1

    if window.is_pressed(ti.ui.RETURN):
        animating = False
    if window.is_pressed(ti.ui.SPACE):
        animating = True

    ext_acc = (ti.math.vec3(0,-9.8,0) + ad * ti.math.vec3(5,0,-5) + ws * ti.math.vec3(-5,0,-5)).normalized() * 9.8
    if animating:
        solver.step_solver(ext_acc)
    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=particle_grid.domain_sz * np.array((0.5, 1, 0.5)), color=(1, 1, 1))
    scene.particles(particle_grid.particle_field.p, radius=particle_grid.particle_radius, per_vertex_color=particle_grid.particle_field.color)
    # scene.particles(particle_grid.particle_field.p, radius=particle_grid.particle_radius, per_vertex_color=solver.solver_particles.dSDF)


    # scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
    canvas.scene(scene)
    if frame_count % 100 == 0:
        print(f"Frame: {frame_count}")
        # print(particle_grid.particle_field.p.to_numpy().max(axis=0))
    frame_count += 1
    window.show()
