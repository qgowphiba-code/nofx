"""Stand-in for PenSimPy's compiled integrator (fastodeint, not on PyPI). Mirrors the simulator's own fast=False branch:
odeint over indpensim_ode_py on t_span = arange(t_start, t_end, h), returning the final state. With PenSimEnv(fast=False)
this module is imported but never called."""
import numpy as np
from scipy.integrate import odeint
from pensimpy.ode.indpensim_ode_py import indpensim_ode_py


def integrate(x0, par, t_start, t_end, h):
    t_span = np.arange(t_start, t_end, h).tolist()
    return odeint(indpensim_ode_py, x0, t_span, tfirst=True, args=(par,))[-1]
