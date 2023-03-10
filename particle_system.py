import taichi as ti
from taichi.math import vec2, vec3, ivec2, ivec3, mat2, mat3
import numpy as np
from functools import reduce


@ti.dataclass
class Material:
    is_liquid: int # 1 if is liquid, 0 if not
    is_dynamic: int # 1 if is dynamic, 0 if not
    mass_per_particle: float
    mass_per_particle_inv: float
    rest_density_inv: float
    color: vec3

# default global parameters
domain_axis_sz = 1.0
particle_radius = 0.01

liquid_volumn_k = 0.8

obj_particle_sort = False

@ti.data_oriented
class ParticleSystem:
    def __init__(self, configs:dict):
        self.configs = configs
        
        # particles used during the simulation
        @ti.dataclass
        class Particle:
            p: vec3 # position
            color: vec3
            material: int
            particle_id: int
            obj_id: int
            SDF: float
            dSDF: vec3
            r: vec3 # used in rigid body constraint

        self.domain_sz = np.array(configs.get("domain_sz", [domain_axis_sz] * 3))

        self.particle_radius = configs.get("particle_radius", particle_radius)
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4
        # self.m_V0 = 0.8 * self.particle_diameter ** 3

        self.materials = Material.field(shape=(len(configs["materials"])))
        particle_volumn = self.particle_diameter ** 3
        self.min_mass = float('inf')
        for i, m in enumerate(configs["materials"]):
            is_liquid = m["is_liquid"]
            particle_mass = particle_volumn * m["density"] * (liquid_volumn_k if is_liquid else 1)
            self.min_mass = min(self.min_mass, particle_mass)
            self.materials[i] = Material(
                is_liquid,
                m.get("is_dynamic", 1) if not is_liquid else 1,
                particle_mass,
                1 / particle_mass,
                1 / m["density"],
                ti.math.vec3(m["color"])
            )

        self.grid_cell_sz = self.support_radius
        self.grid_num = ivec3(np.ceil(self.domain_sz / self.grid_cell_sz).astype(np.int32))
        temp = np.cumprod(self.grid_num)
        self.grid_szs = ivec3(1, *temp[:-1])
        self.num_cells = int(temp[-1])

        self.particle_positions_list = []
        self.materials_list = []
        self.colors_list = []
        self.sdf_list = []
        self.dsdf_list = []
        for i, m in enumerate(configs["objects"]):
            obj_material = m["material"]
            obj_shape = m["shape"]
            pp = np.empty([0, 3], dtype=float)
            pm = np.array([], dtype=int)
            pc = np.array([0, 3], dtype=float)
            if obj_shape == "cube":
                sz = np.array(m["size"])
                if "start" in m:
                    start = np.array(m["start"])
                else:
                    center = np.array(m["center"])
                    start = center - sz / 2
                thickness = m.get("thickness", 0)
                pp, pm, pc, sdf, dsdf = self.generate_particles_for_cube(start, sz, obj_material, thickness)
            elif obj_shape == "sphere":
                center = np.array(m["center"])
                radius = m["radius"]
                thickness = m.get("thickness", 0)
                pp, pm, pc, sdf, dsdf = self.generate_particles_for_sphere(center, radius, obj_material, thickness)
            else:
                print(f"unsupported shape encountered: {obj_shape}")
            self.particle_positions_list.append(pp)
            self.materials_list.append(pm)
            self.colors_list.append(pc)
            self.sdf_list.append(sdf)
            self.dsdf_list.append(dsdf)
        
        self.total_particle_num = reduce(lambda x, y: x+y, (a.shape[0] for a in self.materials_list))
        self.shift = int(np.ceil(np.log2(self.total_particle_num)))
        self.mask = (1 << self.shift) - 1
        self.particle_field = Particle.field(shape=(self.total_particle_num,))
        # for storing particle_id -> particle_field_idx relationship
        # after regional sort, each object region in obj_particle_ids will be sorted for access with less cache miss
        self.obj_particle_ids = ti.field(dtype=int, shape=(self.total_particle_num,))
        self.obj_ids = ti.field(dtype=int, shape=(self.total_particle_num,))
        # variables for counting sort
        self.particle_field_alt = Particle.field(shape=(self.total_particle_num,))
        self.particle_grid_id = ti.field(dtype=int, shape=(self.total_particle_num,))
        self.cell_particle_counts = ti.field(dtype=int, shape=(self.num_cells + 1,))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.num_cells + 1)
        self.solver_particle_registered = False

        idx_base = 0
        self.liquid_regions = []
        self.solid_regions = []
        for oi, (op, om, oc, osdf, odsdf) in enumerate(zip(self.particle_positions_list, self.materials_list, self.colors_list, self.sdf_list, self.dsdf_list)):
            if self.materials[om[0]].is_liquid == 0:
                c = op.mean(axis=0)
                r = op - c
                sdf = osdf
                dsdf = odsdf
            else:
                sdf = np.full(len(op), np.inf, float)
                dsdf = np.zeros((len(op), 3))
                r = np.zeros((len(op), 3))

            self._add_obj(idx_base, oi, len(op), op, om, oc, sdf, dsdf, r)
            if self.materials[om[0]].is_liquid:
                self.liquid_regions.append((idx_base, idx_base + len(op)))
            else:
                self.solid_regions.append((idx_base, idx_base + len(op)))
            idx_base += len(op)
    
    @ti.kernel
    def regional_sort_pre(self):
        for p in self.obj_particle_ids:
            self.obj_particle_ids[p] |= (self.obj_ids[p] << self.shift)
    
    @ti.kernel
    def regional_sort_post(self):
        for p in self.obj_particle_ids:
            self.obj_particle_ids[p] &= self.mask

    def regional_sort(self):
        self.regional_sort_pre()
        ti.algorithms.parallel_sort(self.obj_particle_ids)
        self.regional_sort_post()

    def register_solver_particles(self, solver_particle_type):
        if solver_particle_type is not None:
            self.solver_particles = solver_particle_type.field(shape=(self.total_particle_num,))
            self.solver_particles_alt = solver_particle_type.field(shape=(self.total_particle_num,))
            self.solver_particle_registered = True
        else:
            self.solver_particles = ti.field(int, shape=())
            self.solver_particles_alt = ti.field(int, shape=())
        return self.solver_particles

    @ti.kernel
    def counting_sort_pre(self):
        self.cell_particle_counts.fill(0)
        for p in self.particle_field:
            id = self.pos2idx(self.particle_field[p].p)
            self.particle_grid_id[p] = id
            self.cell_particle_counts[id] += 1
    
    @ti.kernel
    def counting_sort_post(self):
        for p in self.particle_field:
            sorted_pos = ti.atomic_sub(self.cell_particle_counts[self.particle_grid_id[p]], 1) - 1
            self.particle_field_alt[sorted_pos] = self.particle_field[p]
            self.obj_particle_ids[self.particle_field[p].particle_id] = sorted_pos
            if self.solver_particle_registered:
                self.solver_particles_alt[sorted_pos] = self.solver_particles[p]

        for p in self.particle_field:
            self.particle_field[p] = self.particle_field_alt[p]
            if self.solver_particle_registered:
                self.solver_particles[p] = self.solver_particles_alt[p]

    @ti.kernel
    def prefix_sum(self):
        ti.loop_config(serialize=True)
        for c in range(self.num_cells):
            self.cell_particle_counts[c + 1] += self.cell_particle_counts[c]

    def counting_sort(self):
        if not hasattr(self, "solver_particles"):
            print("Particle grid cannot perform any operation untill solver particles get registered")
        self.counting_sort_pre()
        self.prefix_sum_executor.run(self.cell_particle_counts)
        # self.prefix_sum()
        self.counting_sort_post()
        if obj_particle_sort:
            self.regional_sort()

    @ti.func
    def get_grid_idx(self, pos):
        return (pos / self.grid_cell_sz).cast(int)
    
    @ti.func
    def get_flattened_idx(self, idx) -> int:
        return idx.dot(self.grid_szs)
    
    @ti.func
    def pos2idx(self, pos):
        return self.get_flattened_idx(self.get_grid_idx(pos))
    
    @ti.kernel
    def _add_obj(
        self,
        idx_base: int,
        obj_id: int,
        particle_num: int,
        particle_pos: ti.types.ndarray(),
        material_arr: ti.types.ndarray(),
        particle_colors: ti.types.ndarray(),
        SDF: ti.types.ndarray(),
        dSDF: ti.types.ndarray(),
        r: ti.types.ndarray()
        ):
        # print(particle_pos.shape)
        for i in range(idx_base, idx_base + particle_num):
            self.particle_field[i].p = vec3([particle_pos[i - idx_base, j] for j in ti.static(range(3))])
            self.particle_field[i].color = vec3([particle_colors[i - idx_base, j] for j in ti.static(range(3))])
            self.particle_field[i].material = material_arr[i - idx_base]
            self.particle_field[i].particle_id = i
            self.particle_field[i].obj_id = obj_id
            self.particle_field[i].SDF = SDF[i - idx_base]
            self.particle_field[i].dSDF = vec3([dSDF[i - idx_base, j] for j in ti.static(range(3))])
            self.particle_field[i].r = vec3([r[i - idx_base, j] for j in ti.static(range(3))])
            self.obj_particle_ids[i] = i

    def generate_particles_for_cube(self, lower_corner, size, material, thickness):
        slices = tuple(
            slice(
                lower_corner[i] + self.particle_radius,
                lower_corner[i] + size[i] - self.particle_radius,
                self.particle_diameter
                ) for i in range(3))
        particle_positions = np.mgrid[slices].reshape(3, -1).transpose()
        center = np.median(particle_positions, axis=0)
        particle_positions = particle_positions - center + lower_corner + size / 2
        lower_corner = particle_positions.min(axis=0) - self.particle_radius
        size = particle_positions.max(axis=0) - lower_corner + self.particle_radius
        if(thickness>0 and thickness<size.min()/3):
            dist_to_lower = particle_positions - lower_corner
            dist_to_higher = size - dist_to_lower
            dist_to_bounds = np.concatenate((dist_to_lower, dist_to_higher), axis=1)
            mask=(dist_to_bounds<thickness).any(axis=1)
            particle_positions=particle_positions[mask]
        np.random.shuffle(particle_positions)
        print("cube num particles:",len(particle_positions))

        num_new_particles = particle_positions.shape[0]
        material_arr = np.full(num_new_particles, material, dtype=int)
        colors = np.tile(self.materials[material].color, [num_new_particles, 1])

        dist_to_lower = particle_positions - lower_corner
        dist_to_higher = size - dist_to_lower
        dist_to_bounds = np.concatenate((dist_to_lower, dist_to_higher), axis=1)
        dist_idx = np.argmin(dist_to_bounds, axis=1)
        sdf = dist_to_bounds[np.arange(len(dist_idx)), dist_idx]

        min_dists = dist_to_bounds[np.arange(len(dist_idx)), dist_idx]
        temp = (dist_to_bounds == min_dists[:,None]).astype(np.float64)
        dsdf = temp[:,3:] - temp[:,:3]

        # dsdf /= (np.linalg.norm(dsdf, axis=1, keepdims=True) + 1e-16)

        # dsdf = np.zeros((len(dist_idx), 3))
        # dsdf[np.arange(len(dist_idx)),dist_idx % 3] = 1
        # dsdf[dist_idx < 3] *= -1
        return particle_positions, material_arr, colors, sdf, dsdf
    
    def generate_particles_for_sphere(self, center, radius, material, thickness):
        lower_corner = center - radius * 1.2
        size = np.array([2 * radius * 1.2] * 3)
        particle_pos, ma, pc, _, _ = self.generate_particles_for_cube(lower_corner, size, material, 0)
        center = np.median(particle_pos, axis = 0)
        # particle_pos+=np.random.randn(particle_pos.shape[0],particle_pos.shape[1])*self.particle_radius/2
        d = particle_pos - center
        r = np.linalg.norm(d, axis = 1)
        mask = r <= radius 
        if(thickness > 0 and thickness < radius):
            mask &=  r >= (radius-thickness)
        particle_pos = particle_pos[mask]
        ma = ma[mask]
        pc = pc[mask]
        r = r[mask]
        d = d[mask]
        sdf = radius - r
        dsdf = d / r[:,None]
        return particle_pos, ma, pc, sdf, dsdf
    
    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()) -> int:
        center_cell = self.get_grid_idx(self.particle_field[p_i].p)
        result = 0
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            nb = center_cell + offset
            if (0 <= nb).all() and (nb < self.grid_num).all():
                id = self.get_flattened_idx(nb)
                start = self.cell_particle_counts[id]
                end = self.cell_particle_counts[id + 1]
                for p_j in range(start, end):
                    diff = self.particle_field[p_i].p - self.particle_field[p_j].p
                    d2 = diff.dot(diff)
                    if p_i != p_j and d2 < self.support_radius ** 2:
                        task(p_i, p_j, diff, d2, ret)
                        result += 1
        return result
