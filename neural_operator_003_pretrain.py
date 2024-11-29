"""
pre-training a neural operator so that it can be used as an initialization for a downstream task.
To make this principled we should optimize for a different but related task.

In this case we set the target parameters different to training parameters, but in the same physical model class, i.e. fluid-flow.
"""
#%%
from src.pretrain_no import train_model
model, grad_variance_list = train_model()
