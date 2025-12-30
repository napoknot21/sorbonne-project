from __future__ import annotations

import os
import numpy as np

from typing import List, Optional

from src.parameters import u_exact, f_rhs, U_H10_NORM2, MIN_FLOAT_VALUE_NEGATIVE



# Question 1

def FEM_1d_assemble (x : Optional[List[float]] = None) :
    """
    
    """
    x = np.asarray(x, dtype=float)

    if _is_valid_mesh(x) is False :
        return None, None


    N = len(x) - 1

    A = np.zeros((N+1, N+1), dtype=float)
    b = np.zeros(N+1, dtype=float)

    for k in range(1, N+1) :

        h = x[k] - x[k-1]

        # Matrice local
        A[k-1, k-1] += 1/h
        A[k-1, k] += -(1/h)
        A[k, k-1] += -(1/h)
        A[k, k] += 1/h

        # midpoint qudrature for load
        m = 0.5 * (x[k] + x[k-1])
        f_m = f_rhs(m)

        b[k-1] += f_m * (h/2)
        b[k] += f_m * (h/2)
        
    # Interior dofs only Dirichlet 0 at both ends)
    A = A[1:N, 1:N]
    b = b[1:N]

    return A, b



# Question 2

def FEM_1d_solve (x : Optional[List[float]] = None) :
    """
    
    """
    x = np.asarray(x, dtype=float)

    if _is_valid_mesh(x) is False :

        print("\n[-] Error : The input vector x is not a valid mesh")
        return None, None, None
    
    N = len(x) - 1
    A, b = FEM_1d_assemble(x)

    U_int = np.linalg.solve(A, b)

    U_h = np.zeros(N+1, dtype=float)    # size N-1
    U_h[1:N] = U_int                    # size N+1 with boundaries

    return A, b, U_h


# ------- Error in energy norme mesure --------

# Question 3

def FEM_1d (
        
        x : Optional[List[float]] = None,
        compute_l2 : bool = True
        
    ) :
    """
    
    """
    x = np.asarray(x, dtype=float)

    if _is_valid_mesh(x) is False :
        return None, None, None, None, None
    
    N = len(x) - 1

    A, b, U_h = FEM_1d_solve(x)
    U_int = U_h[1:N]

    # Garlekin
    U_h_norm2 = float(U_int.T @ (A @ U_int))
    err2 = U_H10_NORM2 - U_h_norm2

    if err2 < 0 and err2 > MIN_FLOAT_VALUE_NEGATIVE : # -1e-12
        err2 = 0.0

    if err2 < 0 :
        return None, None, None, None, None
    
    err_H10 = np.sqrt(err2)
    err_L2 = None

    if compute_l2 is True :
        err_L2 = L2_error(x, U_h)

    return A, b, U_h, err_H10, err_L2


# Question 4

def mesh_uniform (N) :
    """
    
    """
    # N elements -> N+1 show nod

    return np.linspace(0, 1.0, N+1)



# Question 5



# Question 6

def mesh_geometric (N, alpha) : 
    """
    
    """
    i = np.arange(N+1, dtype=float)
    x = (i / N) ** alpha

    x[0], x[-1] = 0.0, 1.0

    return x




def errors_vs_alpha (N = 50, alphas=alphas)


# ------- Auxiliar functions --------

def _is_valid_mesh (x : Optional[List[float]] = None) -> bool :
    """
    
    """

    if len(x) <= 0 :
        return False
    
    return (x[0] == 0 and x[-1] == 1 and np.all(np.diff(x) > 0))


def L2_error (
    
        x : Optional[List[float]] = None,
        U_h : Optional[List[float]] = None
        
    ) :
    """
    
    """
    x = np.asarray(x, dtype=float)
    U_h = np.asarray(U_h, dtype=float)

    N = len(x) - 1

    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gw = np.array([1.0, 1.0])

    err2 = 0.0

    for k in range(1, N+1) :

        a, b = x[k-1], x[k]
        h = b - a

        mid = 0.5 * (a+b)
        half = 0.5 * h

        UL, UR = U_h[k-1], U_h[k]

        for xi, wi in zip(gp, gw) :

            X = mid + half * xi
            phi_L = (b - x)/h
            phi_R = (X - a)/h

            U_h_x = UL * phi_L + UR * phi_R

            diff = u_exact(X) - U_h_x

            err2 += wi * diff * diff * half

    return np.sqrt(err2)
