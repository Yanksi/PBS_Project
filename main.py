from particle_system import ParticleSystem
from pbf_solver import PBF_Solver
import yaml
import taichi as ti

ti.init(
    arch=ti.gpu
)

with open("pbf_config.yml", "r") as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

particle_grid = ParticleSystem(config)
solver = PBF_Solver(particle_grid, config)

window = ti.ui.Window('PBF3D', res = (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

camera.position(2, 2, 2)
camera.up(0.0, 0.0, 1.0)
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

    if window.is_pressed(ti.ui.SPACE): animating = True

    if particle_grid.dim == 2:
        ext_acc = ti.math.vec2(0, -9.8) + ad * ti.math.vec2(5, 0)
        if animating:
            solver.step_solver(ext_acc)
        canvas.set_background_color((0,0,0))
        canvas.circles(particle_grid.particle_field.p, radius=particle_grid.particle_radius, color=particle_grid.particle_field.color)
    else:
        ext_acc = ti.math.vec3(0,0,-9.8) + ad * ti.math.vec3(5,0,0) + ws * ti.math.vec3(0,5,0)
        if animating:
            solver.step_solver(ext_acc)
        camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        scene.particles(particle_grid.particle_field.p, radius=particle_grid.particle_radius, per_vertex_color=particle_grid.particle_field.color)

        # scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
        canvas.scene(scene)
    if frame_count % 100 == 0:
        print(f"Frame: {frame_count}")
    frame_count += 1
    window.show()
