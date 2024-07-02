from phi.torch.flow import *

velocity = StaggeredGrid(0, x=8, y=8, bounds=Box(x=100, y=100))  # or CenteredGrid(...)
smoke = CenteredGrid(0, ZERO_GRADIENT, x=8, y=8, bounds=Box(x=100, y=100))
INFLOW = 0.2 * resample(Sphere(x=50, y=9.5, radius=5), to=smoke, soft=True)
pressure = None


@jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, s, p, dt=1.):
    s = advect.mac_cormack(s, v, dt) + INFLOW
    buoyancy = resample(s * (0, 0.1), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt
    v, p = fluid.make_incompressible(v, (), Solve(x0=p))
    return v, s, p


velocity, smoke, pressure = step(velocity, smoke, pressure)
velocity, smoke, pressure = step(velocity, smoke, pressure)
velocity, smoke, pressure = step(velocity, smoke, pressure)
velocity, smoke, pressure = step(velocity, smoke, pressure)
velocity, smoke, pressure = step(velocity, smoke, pressure)

print(type(smoke.data))
print(smoke.data.shape)
# print(smoke.data.numpy('y,x'))

print(smoke.data.dimension)
# print(f"{smoke.data:full:shape}")