from typing import List, Sequence
import os
import glob
import numpy as np
from math import sqrt, ceil, floor

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.functional import F

def resolve_path(path, unique=True):
    """
    Resolve a path that may contain variables and user home directory references and globs.
    if "unique" is True, and there are many matches, panic.
    Otherwise return the first/only match.
    """
    path = os.path.expandvars(os.path.expanduser(path))
    matches =  glob.glob(path)
    if unique and len(matches) > 1:
        raise ValueError("Too many matches for glob: {}".format(path))
    return glob.glob(path)[0]

def isscalar(v):
    """
    how is this not a builtin?
    """
    v = torch.as_tensor(v)
    return (
        v.shape == torch.Size([]) or
        v.shape == torch.Size([1]))