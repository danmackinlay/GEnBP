"""
Utilities to convert between different representations of phiflow Fields, phiflow Tensors, pytorch tensors and numpy ndarrays.

This has sprouted many hairs and could be pruned/refactoed
"""
from collections import namedtuple
from phi.flow import *
from phi.field import Field
from phi.math import Shape, extrapolation
from einops import rearrange
import numpy as np
import warnings

def _to_native(t, channels_last=False, force_channel=True):
    """
    copied from `native_call`.

    vector of input tensors are converted to centered native tensors.
    Order depending on `channels_last`.

    Args:
        t: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
    Returns:
        `Tensor` with batch and spatial dimensions of `t` and single channel dimension `vector` if `force_channel`.
    """
    t = as_centered_tensor(t)
    if force_channel:
        t = with_channel(t)
    batch = t.shape.batch
    spatial = t.shape.spatial
    groups = (
        (*batch, *spatial.names, *t.shape.channel)
        if channels_last
        else (*batch, *t.shape.channel, *spatial.names)
    )
    return math.reshaped_native(t, groups)


def to_native(*ts, **kwargs):
    """
    vector of input tensors are converted to centered native tensors depending on `channels_last`.

    Args:
        *ts: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
    """
    return [_to_native(t, **kwargs) for t in ts]


def to_natives(ts, **kwargs):
    """
    Same as `to_native`, but takes a single array argument because that is easier for `map`

    Args:
        ts: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
    """
    return [_to_native(t, **kwargs) for t in ts]


def to_native_chan_last(*t, **kwargs):
    """
    helper function because it's tedious to pass arguments into iterators.
    """
    return to_native(*t, channels_last=True, **kwargs)


def to_natives_chan_last(t, **kwargs):
    """
    Same as `to_native`, but takes a single array argument because that is easier for `map`
    """
    return to_natives(t, channels_last=True, **kwargs)



def wrap_field_like(
        t,
        prototype) -> Field:
    '''
    Turn phi tensor into a centred field
    '''

    return CenteredGrid(
        t,
        bounds=prototype.bounds,
        extrapolation=prototype.extrapolation,
        resolution=prototype.resolution)


def with_channel(t: Tensor)-> Tensor:
    """
    guarantees a channel dimension called vector with size at least 1 for any phiflow tensor
    """
    if len(channel(t.shape)) == 0:
        return math.expand(t, channel(vector=1))
    return t


def as_centered_field(field: Field, extrap=None) -> CenteredGrid:
    '''
    resample the input `Field` and return a corresponding `CenteredGrid`.
    Handy if you want to do something simple with a StaggeredGrid, for example.
    '''
    # `hasattr(field, 'at_centers')` is True even when the method does not exist
    if not isinstance(field, CenteredGrid):
        field = field.at_centers()
    if extrap is not None:
        field = field.with_extrapolation(extrap)
    return field


def as_centered_tensor(t: Tensor or Field) -> Tensor:
    """
    guarantees a phiflow tensor characterising values on a CentredGrid
    """
    if isinstance(t, Field):
        t = as_centered_field(t).values
    return t


# def as_cattable_field(field: Field, extrap=None) -> CenteredGrid:
#     '''
#     possibly resample the input `Field` and return a corresponding `CenteredGrid` with a channel dimension of at least 1
#     '''
#     # `hasattr(field, 'at_centers')` is True even when the method does not exist
#     if not isinstance(field, CenteredGrid):
#         field = as_centered_field(field)
#     if len(channel(field)) == 0:
#         field = math.expand(field, channel(vector=1))
#     if extrap is not None:
#         field = field.with_extrapolation(extrap)
#     return field


# def cat_fields(vs, extrap=None):
#     """
#     Densely packs a list of fields into a single vector CenteredGrid along the channel dimension
#     """
#     return field.concat([
#         as_cattable_field(v, extrap=extrap)
#         for v in vs
#     ], dim=channel('vector'))



# def cat_tensors(ts):
#     """
#     Densely packs a list of tensors into a single Tensor along a channel dimension, creating if needed.
#     """
#     return math.concat([
#         with_channel(t)
#         for t in ts
#     ], dim=channel('vector'))


def copy_field(field: Field) -> Field:
    '''
    Return a (deep) copy of the input field.
    Surely there is a simpler way?
    '''
    return field.__class__(
        values=math.copy(field.values),
        bounds=field.bounds,
        resolution=field.resolution,
        extrapolation=field.extrapolation)


# class TorchWrapper:
#     """
#     Convert a Field to a structured (e.g. x,y,chan) torch tensor and back.
#     There is a scaling param because values might like to be normalized.

#     channels are mandatory and will be imposed on the pytorch native tensor
#     """

#     def __init__(self, prototype, scale=1.0, **coords):
#         self.prototype = prototype
#         self.coords = coords
#         self.scale = scale

#     def wrap(self, tens: Tensor) -> Field:
#         """
#         wrap into a phiflow field
#         """
#         if self.prototype.shape.channel_rank == 0:
#             tens = tens.squeeze(-1)
#         if self.scale == 1.0:
#             return wrap_field_like(tens, self.prototype)
#         return wrap_field_like(
#             tens/self.scale, self.prototype)

#     def unwrap(self, field: Field) -> Tensor:
#         [vector] = to_native_chan_last(
#             field, force_channel=True)
#         if self.scale != 1.0:
#             vector = vector*self.scale
#         return vector


class TorchVectorWrapper:
    """
    Convert a field to a 1d torch tensor and back.
    Batch, if it exists, is first dim.
    There is a scaling param because values might like to be normalized.

    TODO: convert into thin wrapper around TorchWrapper using `vector_packery`
    """

    def __init__(self, prototype, scale=1.0, name="", **coords):
        self.prototype = prototype
        self.coords = coords
        self.scale = scale
        self.name = name  # optionally keep track of which wrapper is which

    def __repr__(self) -> str:
        return (
            f"TorchVectorWrapper(name={self.name},"
            f"shape={self.prototype.non_batch.shape.size},"
            f"scale={self.scale})")

    def wrap(self, tensor: Tensor) -> Field:
        """
        reshape and wrap vector into a phiflow field on a basic CenteredGrid
        """
        if tensor.ndim == 1:
            # unbatched
            tensor = tensor.reshape(self.prototype.shape.non_batch.sizes)
            tensor = math.reshaped_tensor(tensor, self.prototype.shape.non_batch)

        elif tensor.ndim == 2:
            # batched
            tensor = tensor.reshape((-1,) + self.prototype.shape.non_batch.sizes)
            tensor = math.reshaped_tensor(tensor, batch('batch') & self.prototype.shape.non_batch)
        else:
            raise ValueError(
                f"vectors shape {tensor.shape} does not conform to prototype shape self.prototype.shape"
            )

        if self.scale != 1.0:
            tensor /= self.scale

        return wrap_field_like(
            tensor, self.prototype)

    def unwrap(self, field: Field) -> Tensor:
        # optionally resampled StaggeredGrid
        if not isinstance(field, CenteredGrid):
            field = field.at_centers()
        values = field.values

        if values.shape.batch_rank:
            #batched
            vector = math.reshaped_native(
                values, (values.shape.batch, self.prototype.shape.non_batch,))
        else:
            # unbatched
            vector = math.reshaped_native(
                values, (self.prototype.shape.non_batch,))
        if self.scale != 1.0:
            vector = vector*self.scale
        return vector


Vectorified = namedtuple(
    'Vectorified',
    ['init_rand_t',  'simulate_t',  'p_wrapper',  'v_wrapper',  'f_wrapper'])


def vectorify_sim(
        init_rand, simulate,
        grid_size_x, grid_size_y,
        p_scale=1.0, v_scale=1.0, f_scale=1.0):
    """
    automatically translate the phiflow predictors into pytorch tensors.
    It would be more efficient if we did not instantiate here to check dims,
    but this is a one-time cost.

    NB implicitly assumes CenteredGrid.
    """

    particle, velocity, force = init_rand(n_batch=0)

    p_wrapper = TorchVectorWrapper(
        particle,
        name="p",
        x=grid_size_x,
        y=grid_size_y,
        scale=p_scale
    )
    v_wrapper = TorchVectorWrapper(
        velocity,
        name="v",
        x=grid_size_x,
        y=grid_size_y,
        scale=v_scale,
        vector=2,
    )
    f_wrapper = TorchVectorWrapper(
        force,
        name="f",
        x=grid_size_x,
        y=grid_size_y,
        scale=f_scale,
        vector=2,
    )

    def init_rand_t(n_batch):
        """
        the init function for phiflow, but as a pytorch tensor factory
        """
        particle, velocity, force = init_rand(n_batch)
        return p_wrapper.unwrap(particle), v_wrapper.unwrap(velocity), f_wrapper.unwrap(force)

    def simulate_t(particle_t, velocity_t, force_t, *args, **kwargs):
        """
        the simulate function for phiflow, but as a pytorch tensor predictor
        """
        i = 0  #plz brekpoint me
        p_ = p_wrapper.wrap(particle_t)
        v_ = v_wrapper.wrap(velocity_t)
        f_ = f_wrapper.wrap(force_t)
        pred_particle, pred_velocity, pressure = simulate(
            p_, v_, f_,
            *args, **kwargs)
        return p_wrapper.unwrap(
            pred_particle
        ), v_wrapper.unwrap(
            pred_velocity
        ), pressure

    return Vectorified(init_rand_t, simulate_t, p_wrapper, v_wrapper, f_wrapper)


Torchified = namedtuple(
    'Torchified',
    ['init_rand_t',  'simulate_t',  'p_wrapper',  'v_wrapper',  'f_wrapper'])


def torchify_sim(init_rand, simulate, grid_size_x, grid_size_y, p_scale=1.0, v_scale=1.0, f_scale=1.0):
    """
    automatically translate the phiflow predictors into structured pytorch tensors.

    It would be more efficient if we did not instantiate here to check dims,
    but this is a one-time cost.
    """

    particle, velocity, force = init_rand(n_batch=0)

    p_wrapper = TorchWrapper(
        particle,
        x=grid_size_x,
        y=grid_size_y,
        scale=p_scale
    )
    v_wrapper = TorchWrapper(
        velocity,
        x=grid_size_x,
        y=grid_size_y,
        scale=v_scale,
        vector=2,
    )
    f_wrapper = TorchWrapper(
        force,
        x=grid_size_x,
        y=grid_size_y,
        scale=f_scale,
        vector=2,
    )

    def init_rand_t(n_batch):
        """
        the init function for phiflow, but as a pytorch tensor factory
        """
        particle, velocity, force = init_rand(n_batch)
        return p_wrapper.unwrap(particle), v_wrapper.unwrap(velocity), f_wrapper.unwrap(force)

    def simulate_t(particle_t, velocity_t, force_t, *args, **kwargs):
        """
        the simulate function for phiflow, but as a pytorch tensor predictor
        """
        pred_particle, pred_velocity, pressure = simulate(
            p_wrapper.wrap(particle_t),
            v_wrapper.wrap(velocity_t),
            f_wrapper.wrap(force_t), *args, **kwargs)
        return p_wrapper.unwrap(pred_particle), v_wrapper.unwrap(pred_velocity), pressure

    return Torchified(init_rand_t, simulate_t, p_wrapper, v_wrapper, f_wrapper)
