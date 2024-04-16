
import numpy as np


# def multinoise(field_cls, smoothness, smoothness_var, n_batch, **DOMAIN):
#     """
#     factory for functions which run the actual simulation
#     """
#     def init_rand():
#         # Initialization of the particle (i.e. density of the flow) grid with a Phiflow Noise() method
#         field = field_cls(
#             Noise(
#                 batch(batch=n_batch),
#                 scale=scale,
#                 smoothness=smoothness,
#                 vector='x,y'
#             ),
#             extrapolation=getattr(extrapolation, particle_extrapolation),
#             **DOMAIN
#         )
#         return field
#     return init_rand


def taper_func(
        coord,  # <- what even is this? a channel name?
        smooth:float=1,
        epsilon:float=1e-7):
    """
    `smooth` in [0,1]
    """
    # does this acquire the correct phiflow backend?
    from phi import math

    if smooth==0.0:
        # there is a probably a more efficient way
        return math.sum(
            coord * 0,
            math.channel(coord)
        ) + 1
    return (math.prod(
        math.sin(coord.vector[:] *  math.PI),
        math.channel(coord)
    ) + epsilon
    ) ** smooth


def taper(class_, smooth=1.0, *args, **kwargs):
    return class_(
        lambda coord: taper_func(coord, smooth),
        *args,
        **kwargs
    )


def taper_like(prototype, smooth=1.0, *args, **kwargs):
    """
    make a taper function with the same spatial shape as a prototype.
    Assumes field is over a unit box or the coords won't work.
    """
    return taper(
        prototype.__class__,
        smooth=smooth,
        bounds=prototype.bounds,
        extrapolation=prototype.extrapolation,
        resolution=prototype.resolution,)


def ns_sim(
        particle_extrapolation:str='BOUNDARY',
        velocity_extrapolation:str='ZERO',
        NU: float=0.01,
        scale: float= 0.1,
        smoothness: float=3.0,
        # smoothness_var: float=0.0,
        force_scale: float=0.1,
        force_smoothness: float=5.0,
        # force_smoothness_var: float=0.0,
        grid_size=(100,100),
        domain_size=(1.0,1.0),
        force_extrapolation: str='ZERO',
        DT: float=0.01,
        incomp=True,
        backend='torch',
        phi_device='GPU',
        taper_smooth=0.5,
        pos_init=False,   # positive p field, keeps incompressible sane
        staggered=False, # stagger velocities
        jit=True,
        p_noise_power=0.0,
        v_noise_power=0.0,
        advect=True,
        n_skip_steps=1,
    ):
    """
    factory for functions which run the actual simulation
    """
    if backend == 'jax':
        from phi.jax.flow import extrapolation, Box, wrap, advect, diffuse, fluid, math, Solve, CenteredGrid, StaggeredGrid, Noise, batch
    elif backend == 'torch':
        from phi.torch.flow import extrapolation, Box, wrap, advect, diffuse, fluid, math, Solve, CenteredGrid, StaggeredGrid, Noise, batch
        from phi.torch import TORCH
        TORCH.set_default_device(phi_device)
    else:
        from phi.flow import extrapolation, Box, wrap, advect, diffuse, fluid, math, Solve, CenteredGrid, StaggeredGrid, Noise, batch

    grid_size = list(grid_size)
    domain_size = list(domain_size)
    DOMAIN = dict(
        x=grid_size[0],
        y=grid_size[1],
        bounds=Box(
            x=1.0,
            y=1.0
        )
    )

    if staggered:
        vec_grid_cls = StaggeredGrid
    else:
        vec_grid_cls = CenteredGrid

    elem_vol = 1.0 / (grid_size[0] * grid_size[1])
    p_noise_scale = (p_noise_power * elem_vol) ** 0.5 * DT
    v_noise_scale = (v_noise_power * elem_vol) ** 0.5 * DT


    def init_rand(n_batch: int=1):
        # Initialization of the particle (i.e. density of the flow) grid with a Phiflow Noise() method
        batch_chunk = []
        if n_batch is not None and n_batch > 0:
            batch_chunk = [batch(batch=n_batch)]
        particle = CenteredGrid(
            Noise(
                *batch_chunk,
                scale=scale,
                smoothness=smoothness
            ),
            extrapolation=getattr(extrapolation, particle_extrapolation),
            **DOMAIN
        )
        if pos_init:
            particle -= math.min(particle.values*1.01)

        # Initialization of the velocity grid with a Phiflow Noise() method
        velocity = vec_grid_cls(
            Noise(
                *batch_chunk,
                scale=scale,
                smoothness=smoothness,
                vector='x,y'
            ),
            extrapolation=getattr(extrapolation, velocity_extrapolation),
            **DOMAIN
        )

        # Initialization of the force grid.
        force = vec_grid_cls(
            Noise(
                *batch_chunk,
                scale=force_scale,
                smoothness=force_smoothness,
                vector='x,y'
            ),
            extrapolation=getattr(extrapolation, force_extrapolation),
            **DOMAIN
        )

        if taper_smooth> 0.0:
            velocity *= taper_like(velocity, smooth=taper_smooth)
            force *= taper_like(force, smooth=taper_smooth)

        return particle, velocity, force

    def sim_step(
            particle, velocity, force, pressure=None):
        """
        Navier-Stokes Simulation
        cauchy_momentum_step returns velocity and particle by solving cauchy momentum equation for one step.
        cauchy_momentum_solve returns a list of velocity and particle for total simulation time
        Input variables
        velocity, particle : Observed variables (velocity & particle)
        force : External force terms
        **kwargs : Other simulation constraints etc
        """
        # Computing velocity term first
        # Cauchy-momentum equation
        if advect:
            velocity = advect.semi_lagrangian(velocity, velocity, dt=DT) # advection
        velocity = diffuse.explicit(velocity, NU, dt=DT) # diffusion

        # Add external force, constraints
        velocity += DT * particle * force # external force
        # velocity = fluid.apply_boundary_conditions(velocity, obstacles) # obstacles

        # process noise
        if v_noise_scale>0.0:
            velocity += math.random_normal(
                velocity.values.shape) * v_noise_scale

        if incomp:
            # Make incompressible
            # pressure can be returned to accelerate future optimisations by providing a favourable startpoint
            velocity, pressure = fluid.make_incompressible(
                velocity,
                # obstacles,
                solve=Solve(
                    'CG-adaptive',
                    1e-2,
                    0,
                    max_iterations=10000, # high val needed for small grid cells
                    x0=pressure
                ),
            )

        # Computing particle term next
        if p_noise_scale>0.0:
            particle += math.random_normal(
                particle.values.shape) * p_noise_scale
        particle = advect.semi_lagrangian(particle, velocity, dt=DT)

        return particle, velocity, pressure

    if jit:
        init_rand = math.jit_compile(init_rand)
        sim_step = math.jit_compile(sim_step)

    def simulate(particle, velocity, force, pressure=None,  n_skip_steps=n_skip_steps):
        """
        Thin wrapper that runs the sim_step forward multiple steps at once;
        This is faster for incompressible flows because it reycles a guess for the pressure variable, which otherwise must be recomputed for each step.
        """
        for _ in range(n_skip_steps):
            particle, velocity, pressure = sim_step(particle, velocity, force, pressure)
        return particle, velocity, pressure

    return init_rand, simulate


def calc_inv_scale_from_steps(init_rand, sim_step, n_batch=100, n_steps=10, *args, **kwargs):
    """
    Calculate the (inverse) scale of the data for use in scaler
    """
    particle, velocity, force = init_rand(n_batch=n_batch)
    scales = [
        [
            (particle.values**2).mean.sqrt(),
            (velocity.values**2).mean.sqrt(),
            (force.values**2).mean.sqrt()
        ],
    ]
    for i in range(n_steps):
        particle, velocity, pressure = sim_step(
            particle, velocity, force, *args, **kwargs)
        scales.append([
            (particle.values**2).mean.sqrt(),
            (velocity.values**2).mean.sqrt(),
            (force.values**2).mean.sqrt()])
    return np.array(scales).mean(axis=0)


def calc_inv_scale_from_iter(iterator, *args, **kwargs):
    """
    Calculate the (inverse) scale of the data for use in scaler
    """

    scales = []
    for particle, velocity, force in range(iterator):
        scales.append([
            (particle.values**2).mean.sqrt(),
            (velocity.values**2).mean.sqrt(),
            (force.values**2).mean.sqrt()])
    return np.array(scales).mean(axis=0)
