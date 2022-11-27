import taichi as ti
from taichi.math import vec2, vec3, ivec2, ivec3
import numpy as np
from functools import reduce


@ti.dataclass
class Material:
    is_liquid: int # 1 if is liquid, 0 if not
    is_dynamic: int # 1 if is dynamic, 0 if not
    mass_per_particle: float
    rest_density_inv: float
    color: vec3

# default global parameters
domain_axis_sz = 1.0
particle_radius = 0.01

liquid_volumn_k = 1

@ti.data_oriented
class ParticleSystem:
    def __init__(self, configs:dict):
        self.dim = configs["dims"]
        self.configs = configs

        if self.dim == 2:
            self.vec = vec2
            self.ivec = ivec2
        elif self.dim == 3:
            self.vec = vec3
            self.ivec = ivec3
        else:
            print("dimension can only be 2 or 3")
            exit()
        
        # particles used during the simulation
        @ti.dataclass
        class Particle:
            p: self.vec # position
            color: vec3
            material: int

        self.domain_sz = np.array(configs.get("domain_sz", [domain_axis_sz] * self.dim))
        assert(len(self.domain_sz) == self.dim, "Given domain size and dimension does not match!")

        self.particle_radius = configs.get("particle_radius", particle_radius)
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4
        # self.m_V0 = 0.8 * self.particle_diameter ** self.dim

        self.materials = Material.field(shape=(len(configs["materials"])))
        for i, m in enumerate(configs["materials"]):
            if m["is_liquid"]:
                if self.dim == 3:
                    represented_volumn = 4/3 * np.pi * (self.particle_radius * liquid_volumn_k) ** 3 / 0.74
                else:
                    represented_volumn = np.pi * (self.particle_radius * liquid_volumn_k) ** 2 / 0.907
                self.materials[i] = Material(
                    1,
                    1,
                    represented_volumn * m["density"],
                    1 / m["density"],
                    ti.math.vec3(m["color"])
                )
            else:
                represented_volumn = self.particle_diameter ** 3
                self.materials[i] = Material(
                    0,
                    m.get("is_dynamic", 1),
                    represented_volumn * m["density"],
                    1 / m["density"],
                    ti.math.vec3(m["color"])
                )

        self.particle_count = 0

        self.grid_cell_sz = self.support_radius
        self.grid_num = np.ceil(self.domain_sz / self.grid_cell_sz).astype(int)
        temp = np.cumprod(self.grid_num)
        self.grid_szs = self.ivec(1, *temp[:-1])
        self.num_cells = int(temp[-1])

        self.particle_positions_list = []
        self.materials_list = []
        self.colors_list = []
        for i, m in enumerate(configs["objects"]):
            obj_material = m["material"]
            obj_shape = m["shape"]
            if obj_shape == "cube":
                pp, pm, pc = self.generate_particles_for_cube(m["start"], m["size"], obj_material)
                self.particle_positions_list.append(pp)
                self.materials_list.append(pm)
                self.colors_list.append(pc)
            elif obj_shape == "ellip":
                # TODO: implement ellipsoid
                pass
            elif obj_shape.endswith("obj"):
                # TODO: implement obj
                pass
        
        self.total_particle_num = reduce(lambda x, y: x+y, (a.shape[0] for a in self.materials_list))
        self.particle_field = Particle.field(shape=(self.total_particle_num,))
        self.particle_field_alt = Particle.field(shape=(self.total_particle_num,))
        self.particle_grid_id = ti.field(dtype=int, shape=(self.total_particle_num,))
        self.cell_particle_counts = ti.field(dtype=int, shape=(self.num_cells + 1,))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.num_cells + 1)

        idx_base = 0
        for op, om, oc in zip(self.particle_positions_list, self.materials_list, self.colors_list):
            self._add_obj(idx_base, len(op), op, om, oc)
            idx_base += len(op)
        
    
    @ti.kernel
    def counting_sort_pre(self):
        self.cell_particle_counts.fill(0)
        for p in self.particle_field:
            id = self.pos2idx(self.particle_field[p].p)
            self.particle_grid_id[p] = id
            self.cell_particle_counts[id] += 1
    
    @ti.kernel
    def counting_sort_fin(self):
        for p in self.particle_field:
            sorted_pos = ti.atomic_sub(self.cell_particle_counts[self.particle_grid_id[p]], 1)
            self.particle_field_alt[sorted_pos] = self.particle_field[p]
        for p in self.particle_field:
            self.particle_field[p] = self.particle_field_alt[p]

    def counting_sort(self):
        self.counting_sort_pre()
        self.prefix_sum_executor.run(self.cell_particle_counts)
        self.counting_sort_fin()

    @ti.func
    def get_grid_idx(self, pos):
        return (pos / self.grid_cell_sz).cast(int)
    
    @ti.func
    def get_flattened_idx(self, idx):
        return idx.dot(self.grid_szs)
    
    @ti.func
    def pos2idx(self, pos):
        return self.get_flattened_idx(self.get_grid_idx(pos))
    
    @ti.kernel
    def _add_obj(
        self,
        idx_base: int,
        particle_num: int,
        particle_pos: ti.types.ndarray(),
        material_arr: ti.types.ndarray(),
        particle_colors: ti.types.ndarray()
        ):
        # print(particle_pos.shape)
        for i in range(idx_base, idx_base + particle_num):
            self.particle_field[i].p = self.vec([particle_pos[i - idx_base, i] for i in range(self.dim)])
            self.particle_field[i].color = self.vec([particle_colors[i - idx_base, i] for i in range(self.dim)])
            self.particle_field[i].material = material_arr[i - idx_base]

    def generate_particles_for_cube(self, lower_corner, size, material):
        slices = tuple(slice(lower_corner[i], lower_corner[i] + size[i], self.particle_diameter) for i in range(self.dim))
        particle_positions = np.mgrid[slices].reshape(self.dim, -1).transpose()
        num_new_particles = particle_positions.shape[0]
        material_arr = np.full(num_new_particles, material, dtype=int)
        colors = np.tile(self.materials[material].color, [num_new_particles, 1])
        return particle_positions, material_arr, colors
    
    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.get_grid_idx(self.particle_field[p_i].p)
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            nb = center_cell + offset
            if (0 <= nb).all() and (nb < self.grid_num).all():
                id = self.get_flattened_idx(nb)
                start = self.cell_particle_counts[id]
                end = self.cell_particle_counts[id + 1]
                for p_j in range(start, end):
                    diff = self.particle_field[p_i] - self.particle_field[p_j]
                    d2 = diff.dot(diff)
                    if p_i != p_j and d2 < self.support_radius ** 2:
                        task(p_i, p_j, diff, d2, ret)
