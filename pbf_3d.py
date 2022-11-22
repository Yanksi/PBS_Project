import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

# -----PARAMETERS-----

# -WORLD-
dt = 1.0 / 20.0
solve_iteration = 10
res = (500, 500)
world = (20, 20, 20)
boundary = 20
dimension = 3

# -Fluid_Setting-
num_particles = 12000
mass = 1.0
density = 1.0
rest_density = 1.0
padding = 0.4


# -Neighbours_Setting-
h = 1.0
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

# -Confinement/ XSPH Viscosity-
xsph_c = 0.01
vorti_epsilon = 0.01

# -Gradient Approx. delta difference-
g_del = 0.01

# -----FIELDS-----
@ti.dataclass
class Particle3:
    p: ti.math.vec3 # position
    p0: ti.math.vec3 # prev position
    dp: ti.math.vec3 # delta p
    v: ti.math.vec3 # velocity
    vort: ti.math.vec3 # velocity
    l: float # lagrange multiplier
particles=Particle3.field(shape=(num_particles,))
grid_id=ti.field(dtype=int,shape=(num_particles,))
num_particle_in_grid=ti.field(dtype=int,shape=(grid_rows* grid_cols* grid_layers,))
color=ti.Vector.field(3,float,shape=(num_particles,))


# --------------------FUNCTIONS--------------------

prefix_sum=ti.algorithms.PrefixSumExecutor(grid_rows* grid_cols* grid_layers)

@ti.func
def id2color(id):
    c3=id %grid_layers
    rest= id//grid_layers
    c2=rest %grid_cols
    c1=rest//grid_cols
    return ti.math.vec3(c1,c2,c3 )/ti.math.vec3(grid_rows, grid_cols, grid_layers)

@ti.kernel
def pbf_update_grid_id():
    for i in num_particle_in_grid:
        num_particle_in_grid[i]=0
    for i in particles:
        id=get_grid_id(get_grid(particles[i].p))
        grid_id[i]=id
        num_particle_in_grid[id]+=1





@ti.func
def poly6_scalar(d:float)-> float:
    return 315/(64*math.pi*h**9)  * (h**2 - d **2) **3 if 0 <d< h else 0.0


@ti.func
def poly6(dist: ti.template()) ->float:
    sqnorm=dist.dot(dist)
    return 315/(64*math.pi*h**9)  * (h**2 - sqnorm) **3 if 0<sqnorm< h**2 else 0.0


@ti.func
def spiky(dist: ti.template()):
    d = dist.norm()
    return -45.0 / (math.pi*h**6) * (h - d)**2  * dist/d if 0<d<h else ti.Vector( [0.0]*dist.n )



@ti.func
def S_Corr(dist):
    return - S_Corr_k * (poly6(dist)/poly6_scalar(S_Corr_delta_q)) **4



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
    return grid_cord[0]*grid_cols* grid_layers+grid_cord[1]* grid_layers+grid_cord[2]


@ti.kernel
def pbf_solve():
    # ---Calculate lambdas---
    for p in particles.p:
        pos = particles.p[p]
        lower_sum = 0.0
        p_i = 0.0
        spiky_i = ti.Vector([0.0, 0.0, 0.0])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            nb=get_grid(particles.p[p])+offset
            if 0 <= nb[0] <= grid_cols and 0 <= nb[1] <= grid_rows  and 0 <= nb[2] <= grid_layers:
                id=get_grid_id(nb)
                start= num_particle_in_grid[id-1] if id!=0 else 0
                end= num_particle_in_grid[id]
                for i in range(start,end):
                    # ---Poly6---
                    nb_pos = particles.p[i]
                    p_i += mass * poly6(pos - nb_pos)
                    # ---Spiky---
                    s = spiky(pos - nb_pos) / rest_density
                    spiky_i += s
                    lower_sum += s.dot(s)
        constraint = (p_i / rest_density) - 1.0
        lower_sum += spiky_i.dot(spiky_i)
        particles.l[p] = -1.0 * (constraint / (lower_sum + lambda_epsilon))
    # ---Calculate delta P---
    for p in particles.p:
        delta_p = ti.Vector([0.0, 0.0, 0.0])
        pos = particles.p[p]
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            nb=get_grid(particles.p[p])+offset
            if 0 <= nb[0] <= grid_cols and 0 <= nb[1] <= grid_rows  and 0 <= nb[2] <= grid_layers:
                id=get_grid_id(nb)
                start= num_particle_in_grid[id-1] if id!=0 else 0
                end= num_particle_in_grid[id]
                for i in range(start,end):
                    nb_pos = particles.p[i]
                    # ---S_Corr---
                    scorr = S_Corr(pos - nb_pos)
                    left = particles.l[p] + particles.l[i] + scorr
                    right = spiky(pos - nb_pos)
                    delta_p += left * right / rest_density
        particles.dp[p] = delta_p
    # ---Update position with delta P---
    for p in particles.p:
        particles.p[p] += particles.dp[p]


@ti.kernel
def pbf_update():
    for i in particles:
        color[i]=id2color(grid_id[i])
    # ---Update Velocity---
    for v in particles.v:
        particles.v[v] = (particles.p[v] - particles.p0[v]) / dt
    # ---Confinement/ XSPH Viscosity---
    # ---Using wacky gradient approximation for omega---
    for p in particles.p:
        pos = particles.p[p]
        xsph_sum = ti.Vector([0.0, 0.0, 0.0])
        omega_sum = ti.Vector([0.0, 0.0, 0.0])
        # -For Gradient Approx.-
        dx_sum = ti.Vector([0.0, 0.0, 0.0])
        dy_sum = ti.Vector([0.0, 0.0, 0.0])
        dz_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dx_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dy_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dz_sum = ti.Vector([0.0, 0.0, 0.0])
        dx = ti.Vector([g_del, 0.0, 0.0])
        dy = ti.Vector([0.0, g_del, 0.0])
        dz = ti.Vector([0.0, 0.0, g_del])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            nb=get_grid(particles.p[p])+offset
            if 0 <= nb[0] <= grid_cols and 0 <= nb[1] <= grid_rows  and 0 <= nb[2] <= grid_layers:
                id=get_grid_id(nb)
                start= num_particle_in_grid[id-1] if id!=0 else 0
                end= num_particle_in_grid[id]
                for i in range(start,end):
                    nb_pos = particles.p[i]
                    v_ij = particles.v[i] - particles.v[p]
                    dist = pos - nb_pos
                    # ---Vorticity---
                    omega_sum += v_ij.cross(spiky(dist))
                    # -Gradient Approx.-
                    dx_sum += v_ij.cross(spiky(dist + dx))
                    dy_sum += v_ij.cross(spiky(dist + dy))
                    dz_sum += v_ij.cross(spiky(dist + dz))
                    n_dx_sum += v_ij.cross(spiky(dist - dx))
                    n_dy_sum += v_ij.cross(spiky(dist - dy))
                    n_dz_sum += v_ij.cross(spiky(dist - dz))
                    # ---Viscosity---
                    poly = poly6(dist)
                    xsph_sum += poly * v_ij
        # ---Vorticity---
        n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * g_del)
        n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * g_del)
        n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * g_del)
        n = ti.Vector([n_x, n_y, n_z])
        big_n = n.normalized()
        if not omega_sum.norm() == 0.0:
            particles.vort[p] = vorti_epsilon * big_n.cross(omega_sum)
        # ---Viscosity---
        xsph_sum *= xsph_c
        particles.v[p] += xsph_sum

def pbf(ad, ws):
    pbf_pred_pos(ad, ws)
    pbf_update_grid_id()
    ti.algorithms.parallel_sort(grid_id,particles)
    prefix_sum.run(num_particle_in_grid)
    
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
        if window.is_pressed(ti.ui.UP): ws = -1
        if window.is_pressed(ti.ui.DOWN): ws = 1
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
