from __future__ import annotations

import os
import numpy as np

from typing import List, Optional


MIN_FLOAT_VALUE_NEGATIVE = -1e-12

U_H10_NORM2 = 3/5 # ||u||_{H1_O} ^ 2

N_VALUES = [5, 10, 50, 100, 200] # Question 5
ALPHAS = (1, 2, 4, 7, 10) 

N_VALUE = 50
ALPHA_0 = 4


def u_exact (x : Optional[List[float]] = None) :
    """
    
    """
    x = np.asarray(x, dtype=float)
    out = (x ** (3/4)) * (1 -x) 

    return out


def f_rhs (x : Optional[List[float]] = None) :
    """
    
    """
    x = np.asarray(x, dtype=float)
    out = (3/16) * (x ** -(5/4)) * ((7 * x) + 1)

    return out



