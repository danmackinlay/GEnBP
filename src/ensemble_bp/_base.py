

class Converged(Exception):
    pass

VERY_BIG_RANK = 99999


def _factor_name(sim_fn, input_names, output_names):
    return (
        sim_fn.__name__
        + "__" + "_".join(input_names)
        + "__" + "_".join(output_names))
