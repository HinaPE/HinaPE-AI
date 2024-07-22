from phi.torch.flow import *
from phi.field._point_cloud import distribute_points
from tqdm.notebook import trange

domain = Box(x=64, y=64)
obstacle = Box(x=(1, 25), y=(30, 33)).rotated(-20)
initial_particles = distribute_points(union(Box(x=(15, 30), y=(50, 60)), Box(x=None, y=(-INF, 5))), x=64, y=64) * (0, 0)
plot(initial_particles.geometry, obstacle, overlay='args')

