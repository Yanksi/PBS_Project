import taichi as ti
from taichi.math import vec3, mat3
import numpy as np
import math

ti.init(arch=ti.gpu, device_memory_GB=4)

# -----PARAMETERS-----

# -WORLD-
dt = 1.0 / 20.0
solve_iteration = 30
res = (500, 500)
world = (20, 20, 20)
boundary = 20
dimension = 3

# -Fluid_Setting-
num_particles = 22000
mass = 1.0
rest_density = 10.0
rest_density_inv = 1.0 / rest_density
padding = 0.4


# -Neighbours_Setting-
h = 1.0
h2 = h**2
h6 = h**6
h9 = h**9
max_neighbour = 4000

# -Grid_Setting-
grid_rows = int(world[0] / h)
grid_cols = int(world[1] / h)
grid_layers = int(world[2] / h)
max_particle_in_grid = 4000

# -Boundary Epsilon-
b_epsilon = 0.01

# -LAMBDAS-
lambda_epsilon = 100.0

# -S_CORR-
S_Corr_delta_q = 0.3
S_Corr_k = 0.0001
S_Corr_n = 4

# -Confinement/ XSPH Viscosity-
xsph_c = 0.01
vorti_epsilon = 0.01

# -Gradient Approx. delta difference-
g_del = 0.01

# -----FIELDS-----
@ti.dataclass
class Particle3:
    p: vec3 # position
    p0: vec3 # prev position
    dp: vec3 # delta p
    v: vec3 # velocity
    vort: vec3 # velocity
    l: float # lagrange multiplier
particles=Particle3.field(shape=(num_particles,))
grid_id=ti.field(dtype=int,shape=(num_particles,))
particle_id = ti.field(dtype=int, shape=(num_particles, ))
list_curr = ti.field(dtype=int, shape=(grid_rows * grid_cols * grid_layers, ))
num_particle_in_grid=ti.field(dtype=int,shape=(grid_rows * grid_cols * grid_layers + 1,))
color=ti.Vector.field(3,float,shape=(num_particles,))


# --------------------FUNCTIONS--------------------

prefix_sum=ti.algorithms.PrefixSumExecutor(grid_rows * grid_cols * grid_layers)

@ti.func
def id2color(id):
    c3 = id % grid_layers
    rest = id // grid_layers
    c2 = rest % grid_cols
    c1 = rest // grid_cols
    return vec3(c1,c2,c3) / vec3(grid_rows, grid_cols, grid_layers)

@ti.kernel
def pbf_update_grid_id():
    num_particle_in_grid.fill(0)
    for i in particles:
        id = get_grid_id(get_grid(particles[i].p))
        grid_id[i] = id
        num_particle_in_grid[id + 1] += 1


poly6_coeff = 315 / (64 * math.pi * h9)
@ti.func
def poly6(d:vec3) -> float:
    result = 0.0
    d2 = d.dot(d)
    if 0 < d2 < h2:
        result = poly6_nocheck(d2)
    return result

@ti.func
def poly6_nocheck(d2:float) -> float:
    return poly6_coeff * ti.pow(h2 - d2, 3)

# @ti.func
# def poly6(dist: ti.template()) ->float:
#     sqnorm=dist.dot(dist)
#     return 315/(64*math.pi*h**9)  * (h**2 - sqnorm) **3 if 0<sqnorm< h**2 else 0.0

d_spiky_coeff = -45 / (np.pi * h6)
@ti.func
def spiky(d:vec3) -> vec3:
    result = vec3(0)
    dn = d.norm()
    if 0 < dn < h:
        result = spiky_nocheck(d, dn)
    return result

@ti.func
def spiky_nocheck(d:vec3, dn:float) -> vec3:
    return d_spiky_coeff * ti.pow(h - dn, 2) / dn * d

eye3 = mat3([[1,0,0], [0,1,0], [0,0,1]])
@ti.func
def d_spiky(dist:vec3) -> mat3:
    result = mat3(0)
    d2 = dist.dot(dist)
    if 0 < d2 < h2:
        d = ti.sqrt(d2)
        t1 = d_spiky_coeff * ti.pow(h - d, 2) / d
        t2 = d_spiky_coeff * (h2 - d2) / (d2 * d)
        result = t1 * eye3 - t2 * dist.outer_product(dist)
    return result

@ti.func
def cross_mat(v: vec3) -> mat3:
    return mat3([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


poly6_S_Corr = poly6_coeff * (h2 - S_Corr_delta_q**2)**3 if S_Corr_delta_q < h2 else 0.0
S_Corr_coeff = - S_Corr_k / poly6_S_Corr**S_Corr_n
@ti.func
def S_Corr(d: vec3) -> ti.f32:
    return S_Corr_coeff * ti.pow(poly6(d), S_Corr_n)

@ti.func
def S_Corr_nocheck(d2: float) -> ti.f32:
    return S_Corr_coeff * ti.pow(poly6_nocheck(d2), S_Corr_n)


@ti.func
def boundary_condition(v):
    # position filter
    # v is the position in vector form
    lower = padding
    upper = world[0] - padding
    # ---True Boundary---
    for i in ti.static(range(3)):
        v[i]=ti.math.clamp(v[i],lower,upper)
    return v



@ti.func
def get_grid(cord):
    return ti.floor(cord/h,dtype=int)


# --------------------KERNELS--------------------
# avoid nested for loop of position O(n^2)!

@ti.kernel
def pbf_pred_pos(ad: float, ws: float):
    gravity = ti.Vector([0.0, 0.0, -9.8])
    ad_force = ti.Vector([5.0, 0.0, 0.0])
    ws_force = ti.Vector([0.0, 5.0, 0.0])
    for i in particles:
        particles.p0[i] = particles.p[i]
        particles.v[i] += dt * (gravity + ad * ad_force + ws * ws_force + particles.vort[i])
        # ---predict position---
        particles.p[i] = boundary_condition(particles.p[i]+ dt * particles.v[i])
        particles.vort[i]=ti.Vector([0.0, 0.0, 0.0])

@ti.func
def get_grid_id(grid_cord):
    return grid_cord[0] * grid_cols * grid_layers+grid_cord[1] * grid_layers + grid_cord[2]

@ti.kernel
def sort_particles():
    for i in list_curr:
        list_curr[i] = num_particle_in_grid[i]
    for p in particles:
        particle_loc = ti.atomic_add(list_curr[grid_id[p]], 1)
        particle_id[particle_loc] = p

max_constraints = 500
@ti.dataclass
class ActiveNeighbor:
    id:int
    d:vec3
    d2:float
    dn:float
active_constrains = ActiveNeighbor.field()
ti.root.dense(ti.j, max_constraints).dense(ti.i, num_particles).place(active_constrains) 
curr_constraint = ti.field(dtype=int, shape=(num_particles, ))

@ti.kernel
def pbf_solve():
    curr_constraint.fill(0)
    # for p in particles.p:
    for pid in particle_id:
        p = particle_id[pid]
        pos = particles.p[p]
        lower_sum = 0.0
        p_i = 0.0
        spiky_i = ti.Vector([0.0, 0.0, 0.0])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            nb = get_grid(particles.p[p])+offset
            if 0 <= nb[0] <= grid_cols and 0 <= nb[1] <= grid_rows and 0 <= nb[2] <= grid_layers:
                id = get_grid_id(nb)
                start = num_particle_in_grid[id]
                end = num_particle_in_grid[id + 1]
                for i in range(start,end):
                    # ---Poly6---
                    nb_pos = particles.p[particle_id[i]]
                    d = pos - nb_pos
                    d2 = d.dot(d)
                    if 0 < d2 < h2:
                        # constraint_loc = ti.atomic_add(curr_constraint[p], 1)
                        active_constrains[p, curr_constraint[p]] = ActiveNeighbor(particle_id[i], d, d2, ti.sqrt(d2))
                        curr_constraint[p] += 1
    # ---Calculate lambdas---
    for p in particles:
        lower_sum = 0.0
        p_i = 0.0
        spiky_i = ti.Vector([0.0, 0.0, 0.0])
        for c in range(curr_constraint[p]):
            active_nb = active_constrains[p, c]
            nb_id = active_nb.id
            p_i += mass * poly6_nocheck(active_nb.d2)
            # ---Spiky---
            s = spiky_nocheck(active_nb.d, active_nb.dn) * rest_density_inv
            spiky_i += s
            lower_sum += s.dot(s)
        constraint = (p_i * rest_density_inv) - 1.0
        lower_sum += spiky_i.dot(spiky_i)
        particles.l[p] = -1.0 * (constraint / (lower_sum + lambda_epsilon))
    # ---Calculate delta P---
    # for p in particles.p:
    for p in particles:
        delta_p = ti.Vector([0.0, 0.0, 0.0])
        for c in range(curr_constraint[p]):
            active_nb = active_constrains[p, c]
            nb_id = active_nb.id
            # ---S_Corr---
            scorr = S_Corr_nocheck(active_nb.d2)
            left = particles.l[p] + particles.l[nb_id] + scorr
            right = spiky_nocheck(active_nb.d, active_nb.dn)
            delta_p += left * rest_density_inv * right
        particles.dp[p] = delta_p
    # ---Update position with delta P---
    for p in particles.p:
        particles.p[p] += particles.dp[p]


@ti.kernel
def pbf_update():
    for i in particles:
        color[i] = id2color(grid_id[i])
    # ---Update Velocity---
    for v in particles.v:
        particles.v[v] = (particles.p[v] - particles.p0[v]) / dt
    # ---Confinement/ XSPH Viscosity---
    # for p in particles.p:
    for pid in particle_id:
        p = particle_id[pid]
        pos = particles.p[p]
        xsph_sum = vec3(0)
        omega_sum = vec3(0)
        d_omega_p = mat3(0)
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            nb = get_grid(particles.p[p])+offset
            if 0 <= nb[0] <= grid_cols and 0 <= nb[1] <= grid_rows  and 0 <= nb[2] <= grid_layers:
                id = get_grid_id(nb)
                start = num_particle_in_grid[id]
                end = num_particle_in_grid[id + 1]
                for i in range(start,end):
                    nb_pos = particles.p[particle_id[i]]
                    v_ij = particles.v[particle_id[i]] - particles.v[p]
                    dist = pos - nb_pos
                    # ---Vorticity---
                    omega_sum += v_ij.cross(spiky(dist))
                    d_omega_p += cross_mat(v_ij) @ d_spiky(dist)
                    # ---Viscosity---
                    poly = poly6(dist)
                    xsph_sum += poly * v_ij
        # ---Vorticity---
        omega = omega_sum.normalized()
        n = d_omega_p @ omega
        big_n = n.normalized()
        if not omega_sum.norm() == 0.0:
            particles.vort[p] = vorti_epsilon * big_n.cross(omega_sum)
        # ---Viscosity---
        xsph_sum *= xsph_c
        particles.v[p] += xsph_sum

def pbf(ad, ws):
    pbf_pred_pos(ad, ws)
    pbf_update_grid_id()
    prefix_sum.run(num_particle_in_grid)
    sort_particles()
    # ti.algorithms.parallel_sort(grid_id,particles)
    for _ in range(solve_iteration):
        pbf_solve()
    pbf_update()


@ti.kernel
def init():
    for i in particles.p:
        pos_x = 2 + 0.8 * (i % 20)+ ti.random() * b_epsilon
        pos_y = 2 + 0.8 * ((i % 400) // 20)+ ti.random() * b_epsilon
        pos_z = 1 + 0.8 * (i // 400)+ ti.random() * b_epsilon
        particles.p[i] = ti.Vector([pos_x, pos_y, pos_z])
        particles.vort[i] = ti.Vector([0.0, 0.0, 0.0])
        particle_id[i] = i

def main():
    init()
    prefix = "./3d_ply/a.ply"
    window = ti.ui.Window('PBF3D', res = (1024, 1024))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(9,-24,33)
    camera.lookat(9,-24+0.8,33-0.6)
    # gui = ti.GUI('PBF3D', res)
    frame_count = 0
    gui = window.get_gui()
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        gui.text(str(camera.curr_position)+'\n'+str(camera.curr_lookat-camera.curr_position)+'\n'+str(camera.curr_up))
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.particles(particles.p, color = (0.19, 0.26, 0.68),per_vertex_color=color, radius = 0.2)
        # ---Control Waves---
        ad = 0.0
        ws = 0.0
        window.get_event(ti.ui.PRESS)
        if window.event is not None: pass
        if window.is_pressed(ti.ui.LEFT): ad = -1
        if window.is_pressed(ti.ui.RIGHT): ad = 1
        if window.is_pressed(ti.ui.UP): ws = 1
        if window.is_pressed(ti.ui.DOWN): ws = -1
        pbf(ad, ws)

        canvas.scene(scene)
        window.show()
        # ---Frame Control---
        if frame_count % 100 == 0:
            print(f"Frame: {frame_count}")
        frame_count += 1
    return 0


if __name__ == '__main__':
    main()
