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
solver.step_solver(ti.math.vec3([0,0,-9.8]))
