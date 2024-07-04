from phi.torch.flow import *
import pylab

# closed domain
INFLOW_LOCATION = tensor([(12, 4), (13, 6), (14, 5), (16, 5)], batch('inflow_loc'), channel(vector="x,y"))
INFLOW = (1. / 3.) * CenteredGrid(Sphere(center=INFLOW_LOCATION, radius=3), extrapolation.BOUNDARY, x=32, y=40,
                                  bounds=Box(x=(0, 32), y=(0, 40)))
BND = extrapolation.ZERO

smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=(0, 32), y=(0, 40)))  # sampled at cell centers
velocity = StaggeredGrid(0, BND, x=32, y=40,
                         bounds=Box(x=(0, 32), y=(0, 40)))  # sampled in staggered form at face centers


def step(smoke, velocity):
    smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
    buoyancy_force = (smoke * (0, 1)).at(velocity)
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
    velocity, _ = fluid.make_incompressible(velocity)
    return smoke, velocity


for _ in range(20):
    smoke, velocity = step(smoke, velocity)

# store and show final states (before optimization)
smoke_final = smoke
fig, axes = pylab.subplots(1, 4, figsize=(10, 6))
for i in range(INFLOW.shape.get_size('inflow_loc')):
    axes[i].imshow(smoke_final.values.numpy('inflow_loc,y,x')[i, ...], origin='lower', cmap='magma')
    axes[i].set_title(f"Inflow {INFLOW_LOCATION.numpy('inflow_loc,vector')[i]}" + (", Reference" if i == 3 else ""))
pylab.tight_layout()
# pylab.show()


initial_smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=(0, 32), y=(0, 40)))
initial_velocity = StaggeredGrid(math.zeros(batch(inflow_loc=4)), extrapolation.ZERO, x=32, y=40,
                                 bounds=Box(x=(0, 32), y=(0, 40)))
print("Velocity dimensions: " + format(initial_velocity.shape))


def simulate(in_smoke: CenteredGrid, in_velocity: StaggeredGrid):
    for _ in range(20):
        in_smoke, in_velocity = step(in_smoke, in_velocity)

    loss = field.l2_loss(in_smoke - field.stop_gradient(in_smoke.inflow_loc[-1]))
    # optionally, use smoother loss with diffusion steps - no difference here, but can be useful for more complex cases
    # loss = field.l2_loss(diffuse.explicit(smoke - field.stop_gradient(smoke.inflow_loc[-1]), 1, 1, 10))

    return loss, in_smoke, in_velocity


sim_grad = field.functional_gradient(simulate, wrt=[1], get_output=False)
(velocity_grad,) = sim_grad(initial_smoke, initial_velocity)

print("Some gradient info: " + format(velocity_grad))
print(format(velocity_grad.values.inflow_loc[0].vector[
                 0]))  # one example, location 0, x component, automatically prints size & content range

# neat phiflow helper function:
vis.plot(field.vec_length(velocity_grad))  # show magnitude
# vis.show()

sim_grad_wloss = field.functional_gradient(simulate, wrt=[1], get_output=True)  # if we need outputs...
LR = 1e-03
for optim_step in range(80):
    (loss, _smoke, _velocity), (velocity_grad,) = sim_grad_wloss(initial_smoke, initial_velocity)
    initial_velocity = initial_velocity - LR * velocity_grad
    if optim_step < 3 or optim_step % 10 == 9: print(
        'Optimization step %d   , loss: %f' % (optim_step, np.sum(loss.numpy())))

fig, axes = pylab.subplots(1, 3, figsize=(10, 4))
for i in range(INFLOW.shape.get_size('inflow_loc') - 1):
    im = axes[i].imshow(initial_velocity.staggered_tensor().numpy('inflow_loc,y,x,vector')[i, ..., 0], origin='lower',
                        cmap='magma')
    axes[i].set_title(f"Ini. vel. X {INFLOW_LOCATION.numpy('inflow_loc,vector')[i]}")
    pylab.colorbar(im, ax=axes[i])
pylab.tight_layout()
# pylab.show()

fig, axes = pylab.subplots(1, 3, figsize=(10, 4))
for i in range(INFLOW.shape.get_size('inflow_loc') - 1):
    im = axes[i].imshow(initial_velocity.staggered_tensor().numpy('inflow_loc,y,x,vector')[i, ..., 1], origin='lower',
                        cmap='magma')
    axes[i].set_title(f"Ini. vel. Y {INFLOW_LOCATION.numpy('inflow_loc,vector')[i]}")
    pylab.colorbar(im, ax=axes[i])
pylab.tight_layout()
# pylab.show()

smoke = initial_smoke
velocity = initial_velocity

for _ in range(20):
    smoke, velocity = step(smoke, velocity)

fig, axes = pylab.subplots(1, 4, figsize=(10, 6))
for i in range(INFLOW.shape.get_size('inflow_loc')):
    axes[i].imshow(smoke_final.values.numpy('inflow_loc,y,x')[i, ...], origin='lower', cmap='magma')
    axes[i].set_title(f"Inflow {INFLOW_LOCATION.numpy('inflow_loc,vector')[i]}" + (", Reference" if i == 3 else ""))
pylab.tight_layout()
# pylab.show()

fig, axes = pylab.subplots(2, 3, figsize=(10, 6))
for i in range(INFLOW.shape.get_size('inflow_loc') - 1):
    axes[0, i].imshow(
        smoke_final.values.numpy('inflow_loc,y,x')[i, ...] - smoke_final.values.numpy('inflow_loc,y,x')[3, ...],
        origin='lower', cmap='magma')
    axes[0, i].set_title(f"Org. diff. {INFLOW_LOCATION.numpy('inflow_loc,vector')[i]}")
    pylab.colorbar(im, ax=axes[0, i])
for i in range(INFLOW.shape.get_size('inflow_loc') - 1):
    axes[1, i].imshow(smoke.values.numpy('inflow_loc,y,x')[i, ...] - smoke_final.values.numpy('inflow_loc,y,x')[3, ...],
                      origin='lower', cmap='magma')
    axes[1, i].set_title(f"Result {INFLOW_LOCATION.numpy('inflow_loc,vector')[i]}")
    pylab.colorbar(im, ax=axes[1, i])
pylab.show()
