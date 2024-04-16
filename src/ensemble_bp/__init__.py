"""
Implementation of ensemble message passing.

Loosely inspired by Ortiz

* https://colab.research.google.com/drive/1-nrE95X4UC9FBLR0-cTnsIP_XhA_PZKW
* https://github.com/joeaortiz/gbp/blob/master/gbp/gbp.py

see also
https://github.com/krashkov/Belief-Propagation/blob/master/4-ImplementationBP.ipynb

There are several data structure of note:

1. ensembles are batches of vectors, i.e. matrices
2. methods that end in `_d` operator on dictionaries of ensembles
3. beliefs are 2-tuples of (e, prec) pairs, where e is an information vector
   and prec is a precision operator, presumably low-rank or what are we even
   doing?

gamma2 controls the implied noise when calculating ensemble statistics, which propagates through the network as diagonal uncertainty,

sigma2 controls the weighting applied to observations when calculating observation updates.

eta2 reduces the certainty of our ensemble conformation updates.
It defaults to the same as gamma2.
"""

from .factor_graph import FactorGraph