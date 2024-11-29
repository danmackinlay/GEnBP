

class Converged(Exception):
    pass

VERY_BIG_RANK = 99999


def _factor_name(sim_fn, input_names, output_names):
    fn_name = getattr(sim_fn, "__name__", str(sim_fn or ""))
    return fn_name + "__" + "_".join(input_names) + "__" + "_".join(output_names)


def var_dims_d(var_nodes):
    """
    return the dimensions of each implicated var node
    """
    var_dims = {}
    for k, v in var_nodes.items():
        var_dims[k] = v.get_dim()
    return var_dims


def var_slices_d(var_nodes):
    """
    return slices for each var node
    """
    var_slices = {}
    offset = 0
    for k, v in var_nodes.items():
        next_offset = offset + v.get_dim()
        var_slices[k] = slice(offset, next_offset)
        offset = next_offset
    return var_slices


def slice_by_var_d(var_nodes, mat):
    """
    Utility which returns a dict of views of a large matrix with slices
    corresponding to the var nodes.
    This should turn a matrix into a dict of matrices.
    """
    slices = var_slices_d(var_nodes)
    return {k: mat[slices[k], :] for k in slices}
