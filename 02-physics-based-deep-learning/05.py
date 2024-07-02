from phi.torch.flow import *

velocity = StaggeredGrid((0, 0, 0), 0, x=50, y=50, z=50, bounds=Box(x=1, y=1, z=1))  # or CenteredGrid(...)
smoke = CenteredGrid(0, ZERO_GRADIENT, x=50, y=50, z=50, bounds=Box(x=1, y=1, z=1))
INFLOW = 0.5 * resample(Sphere(x=0.5, y=0.5, z=0.5, radius=0.1), to=smoke, soft=True)
pressure = None


# @jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, s, p, dt=1.):
    s = advect.mac_cormack(s, v, dt) + INFLOW
    buoyancy = resample(s * (0, 0, 0.1), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt
    v, p = fluid.make_incompressible(v, (), Solve('auto', 1e-5, x0=p))
    return v, s, p


# velocity, smoke, pressure = step(velocity, smoke, pressure)
#
# d_n = smoke.data.native('x,y,z')
# v_xn = velocity.vector['x'].data.native('x,y,z')
# v_yn = velocity.vector['y'].data.native('x,y,z')
# v_zn = velocity.vector['z'].data.native('x,y,z')
#
# DEBUG = True
#
# if DEBUG:
#     print(smoke.data.shape)
#     print(type(smoke.data))
#     print(type(d_n))
#     print(d_n.size())
#
#     print(velocity.data.shape)
#     print(type(velocity.data))
#     print(type(v_xn))
#     print(type(v_yn))
#     print(type(v_zn))
#     print(v_xn.size())
#     print(v_yn.size())
#     print(v_zn.size())

for _ in view(smoke, velocity, 'pressure', play=False, namespace=globals()).range(warmup=1):
    velocity, smoke, pressure = step(velocity, smoke, pressure)
