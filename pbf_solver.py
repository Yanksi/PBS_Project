import taichi as ti
from taichi.math import vec2, vec3, ivec2, ivec3, mat3
from particle_system import ParticleSystem
import math
import numpy as np

# default solver parameters
S_Corr_delta_q = 0.3
S_Corr_k = 0.0001
S_Corr_n = 4

dt = 0.05

lambda_epsilon = 100.0

@ti.data_oriented
class PBF_Solver:
    def __init__(self, particle_grid: ParticleSystem, config):
        self.particle_grid = particle_grid
        self.dim = self.particle_grid.dim
        self.world_sz = self.particle_grid.domain_sz
        num_particles = self.particle_grid.total_particle_num
        self.vec = self.particle_grid.vec
        self.particles = self.particle_grid.particle_field
        self.materials = self.particle_grid.materials
        @ti.dataclass
        class SolverParticle:
            v: self.vec # velocity
            p0: self.vec # previous position
            dp: self.vec # delta p
            l: float # lagrange multiplier
        self.solver_particles = SolverParticle.field(shape=(num_particles,))
        # if self.dim == 3:
        self.vorticity = ti.Vector.field(3, float, shape=(num_particles,))
        self.h = self.particle_grid.support_radius
        self.padding = config.get("padding", self.h)
        self.d_spiky_coeff = -45 / (math.pi * self.h ** 6)
        self.poly6_coeff = 315 / (64 * math.pi * self.h ** 9)
        S_Corr_delta_q = config["solver"].get("S_Corr_delta_q", S_Corr_delta_q) * self.h
        poly6_S_Corr = self.poly6_coeff * (self.h ** 2 - S_Corr_delta_q ** 2) ** 3
        S_Corr_k = config["solver"].get("S_Corr_k", S_Corr_k)
        self.S_Corr_n = config["solver"].get("S_Corr_n", S_Corr_n)
        self.S_Corr_coeff = - S_Corr_k / poly6_S_Corr ** self.S_Corr_n
        self.dt = config.get("dt", dt)
        self.lambda_epsilon = config["solver"].get("lambda_epsilon", lambda_epsilon)

        # used to get rid of compilation error in 2D mode
        dummy = np.eye(3)
        if self.dim == 2:
            dummy = dummy[:,:2]
        self.dummy_mat = ti.Matrix(dummy)
        self.dummy_matT = self.dummy_mat.transpose()


    @ti.func
    def poly6(self, d2: float) -> float:
        return self.poly6_coeff * ti.pow(self.h ** 2 - d2, 3)
    
    @ti.func
    def d_spiky(self, d, dn):
        return self.d_spiky_coeff * ti.pow(self.h - dn, 2) / dn * d

    @ti.func
    def dd_spiky(self, dist: vec3) -> mat3:
        eye3 = mat3([[1,0,0], [0,1,0], [0,0,1]])
        result = mat3(0)
        d2 = dist.dot(dist)
        if 0 < d2 < self.h**2:
            d = ti.sqrt(d2)
            t1 = self.d_spiky_coeff * ti.pow(self.h - d, 2) / d
            t2 = self.d_spiky_coeff * (self.h**2 - d2) / (d2 * d)
            result = t1 * eye3 - t2 * dist.outer_product(dist)
        return result
    
    @ti.func
    def cross_mat(v: vec3) -> mat3:
        return mat3([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @ti.func
    def S_Corr(self, d2: float) -> float:
        return self.S_Corr_coeff * ti.pow(self.poly6(d2), self.S_Corr_n)   
    
    @ti.func
    def clip_boundary(self, position):
        lower = self.vec(self.padding)
        upper = self.world_sz - self.padding
        return max(min(position, upper), lower)
    

    @ti.kernel
    def advect(self, external_acc: ti.Vector):
        for p in self.particles:
            if self.materials[self.particles[p].material].is_dynamic:
                self.solver_particles[p].p0 = self.particles[p].p
                delta_v = dt * external_acc
                if self.dim == 3:
                    delta_v += dt * self.dummy_matT @ self.vorticity[p]
                    self.vorticity[p] = vec3(0)
                self.solver_particles[p].v += delta_v
                self.particles[p].p = self.clip_boundary(self.particles[p].p + dt * self.solver_particles[p].v)
    
    @ti.func
    def solve_task_lambda(self, pid, pjd, dist, d2, ret:ti.template()):
        # TODO: consider for case which the mateiral of the given two particles are different
        mass_pj = self.materials[self.particles[pjd].material].mass_per_particle
        material_pi = self.materials[self.particles[pid].material]
        # mass_pi = material_pi.mass_per_particle
        dens_pi_inv = material_pi.rest_density_inv
        ret[0] += mass_pj * self.poly6(d2)

        d = ti.sqrt(d2)
        s = self.d_spiky(dist, d) * dens_pi_inv
        ret[1] += s
        ret[2] += s.dot(s)

    @ti.func
    def solve_task_delta_p(self, pid, pjd, dist, d2, ret:ti.template()):
        # TODO: consider for case which the mateiral of the given two particles are different
        material_pi = self.materials[self.particles[pid].material]
        # mass_pi = material_pi.mass_per_particle
        dens_pi_inv = material_pi.rest_density_inv
        scorr = self.S_Corr(d2)
        left = self.solver_particles[pid].l + self.solver_particles[pjd].l + scorr
        d = ti.sqrt(d2)
        right = self.d_spiky(dist, d)
        ret += left * dens_pi_inv * right

    @ti.kernel
    def solve(self):
        for pid in self.particles:
            dens_pi_inv = self.materials[self.particles[pid].material].rest_density_inv
            lower_sum = 0.0
            p_i = 0.0
            d_spiky_i = self.vec(0)
            self.particle_grid.for_all_neighbors(pid, self.solve_task_lambda, [lower_sum, p_i, d_spiky_i])
            constraint = (p_i * dens_pi_inv) - 1.0
            lower_sum += d_spiky_i.dot(d_spiky_i)
            self.solver_particles[pid].l = -1.0 * (constraint / (lower_sum + self.lambda_epsilon))
        
        for pid in self.particles:
            dp = self.vec(0)
            self.particle_grid.for_all_neighbors(pid, self.solve_task_delta_p, dp)
            self.solver_particles[pid].dp = dp
        
        for p in self.particles:
            if self.materials[self.particles[p].material].is_dynamic:
                self.particles[p].p += self.solver_particles[p].dp
    
    @ti.kernel
    def finalize_step(self):
        # TODO
        pass
