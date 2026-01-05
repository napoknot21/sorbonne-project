from __future__ import annotations

import os
import numpy as np

from typing import List, Optional, Tuple

from src.params import u_exact, f_rhs, U_H10_NORM2, MIN_FLOAT_VALUE_NEGATIVE, N_VALUES, N_VALUE, ALPHAS, ALPHA_0
from src.utils import plot_convergence, plot_meshes, plot_alphas, plot_uniform_vs_geometric


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

        # Matrice locale
        A[k-1, k-1] += 1/h
        A[k-1, k] += -(1/h)
        A[k, k-1] += -(1/h)
        A[k, k] += 1/h

        # midpoint quadrature for load
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

    if A is None or b is None or U_h is None :
        return None, None, None, None, None

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

def run_uniform (N_values : Optional[list[int]] = None) :
    """
    Docstring for convergence_uniform
    
    :param N_values: Description
    :type N_values: Optional[list[int]]
    """
    N_values = N_VALUES if N_values is None else N_values

    err_H = []
    err_L2 = []

    for N in N_values :

        x = mesh_uniform(N)
        _, _, _, e_H, e_L2 = FEM_1d(x, True)

        err_H.append(e_H)
        err_L2.append(e_L2)

    return err_H, err_L2


def estimate_slope (N_list, err_list) :
    """
    Docstring for estimate_slope
    
    :param N_list: Description
    :param err_list: Description
    """
    N_arr = np.asarray(N_list, dtype=float)
    e_arr = np.asarray(err_list, dtype=float)

    X = np.log(N_arr)
    Y = np.log(e_arr)

    p, c = np.linalg.lstsq(np.vstack([X, np.ones_like(X)]).T, Y, rcond=None)[0]

    return p


def graph_n_estimation_uniform (N_values : Optional[List[int]] = None) :
    """
    Docstring for graph_n_estimation
    
    :param N_values: Description
    :type N_values: Optional[List[int]]
    """
    N_values = N_VALUES if N_values is None else N_values
    errH_u, errL2_u = run_uniform(N_values)

    print("\n[*] Uniform mesh :")

    for N, e_H, e_L2 in zip(N_values, errH_u, errL2_u) :
        
        print(f"\n[*] N={N:4d}    err_H10={e_H:.6e}  err_L2={e_L2:.6e}")

    print("\n[*] Estimated Slope H10: ", estimate_slope(N_values, errH_u))
    print("\n[*] Estimated Slope L2: ", estimate_slope(N_values, errL2_u))

    plot_convergence(N_values, errH_u, "Uniform mesh: energy error", "||u - uh||_{H1_0}")
    plot_convergence(N_values, errL2_u, "Uniform mesh: L2 error", "||u - uh||_{L2}")

    return None


# Question 6

def mesh_geometric (N, alpha) : 
    """
    
    """
    i = np.arange(N+1, dtype=float)
    x = (i / N) ** alpha

    x[0], x[-1] = 0.0, 1.0

    return x


def mesh_trace (N : Optional[int] = None, alphas : Optional[Tuple] = None) :
    """
    Docstring for mesh_trace
    
    :param N: Description
    :param alphas: Description
    """
    N = N_VALUE if N is None else N
    alphas = ALPHAS if alphas is None else alphas

    plot_meshes(N, alphas, mesh_geometric)

    return None


# Question 7

def run_geometric (N_values, alpha) :
    """
    Docstring for run_geometric
    
    :param N_values: Description
    :param alpha: Description
    """
    err_H = []
    err_L2 = []

    for N in N_values :

        x = mesh_geometric(N, alpha)
        _, _, _, e_H, e_L2 = FEM_1d(x, True)

        err_H.append(e_H)
        err_L2.append(e_L2)

    return err_H, err_L2


def graph_n_estimation_geometric (N_values : Optional[List[int]] = None, alpha : Optional[int] = None) :
    """
    Docstring for graph_n_estimation
    
    :param N_values: Description
    :type N_values: Optional[List[int]]
    """
    N_values = N_VALUES if N_values is None else N_values
    alpha = ALPHA_0 if alpha is None else alpha # Here alpha is := 4

    errH_g, errL2_g = run_geometric(N_values, alpha)

    print(f"\n[*] Geometric mesh (alpha={alpha})")

    for N, e_H, e_L2 in zip(N_values, errH_g, errL2_g) :
        
        print(f"\n[*] N={N:4d}    err_H10={e_H:.6e}  err_L2={e_L2:.6e}")

    print("\n[*] Estimated Slope H10: ", estimate_slope(N_values, errH_g))
    print("\n[*] Estimated Slope L2: ", estimate_slope(N_values, errL2_g))

    plot_convergence(N_values, errH_g, f"Geometric mesh (Alpha={alpha})", "||u - uh||_{H1_0}")
    plot_convergence(N_values, errL2_g, f"Geometric mesh (Alpha={alpha})", "||u - uh||_{L2}")

    return None


# Question 8

def errors_for_multiple_alpha (N : Optional[int] = None, alphas : Optional[List[int]] = None) :
    """
    Question 8 (fixed N):
    - Compute errors for alpha in ALPHAS with N = N_VALUE
    - Choose the best alpha (min energy error)
    - Plot error vs alpha
    
    :param N_values: Description
    :param alphas: Description
    """
    N = N_VALUE if N is None else N
    alphas = ALPHAS if alphas is None else alphas

    err_H = []
    err_L2 = []

    print(f"\n[*] Question 8 : Errors vs alpha (fixed N={N})")

    for a in alphas :

        x = mesh_geometric(N, a)
        _, _, _, eH, eL2 = FEM_1d(x, compute_l2=True)

        err_H.append(eH)
        err_L2.append(eL2)

    best_idx = int(np.argmin(err_H))
    best_alpha = list(alphas)[best_idx]

    print(f"\n[*] Best alpha for fixed N={N} (min energy error): {best_alpha}")
    
    return best_alpha, err_H, err_L2


def graph_alpha_comparaison (N : Optional[int] = None, alphas : Optional[List[int]] = None) :
    """
    """
    N = N_VALUE if N is None else N
    alphas = ALPHAS if alphas is None else alphas
    
    best_alpha, err_H, err_L2 = errors_for_multiple_alpha(N, alphas)

    for a, eH, eL2 in zip(alphas, err_H, err_L2) :
        print(f"\n[*] alpha={a:>2}  err_H10={eH:.6e}  err_L2={eL2:.6e}")

    plot_alphas(N, alphas, err_H, err_L2, "H1_O Error", "L2 Error")

    return None


# Question 9

def compare_uniform_vs_geometric (N : Optional[int] = None, alpha : Optional[int] = None) :
    """
    Docstring for compare_uniform_vs_geometric
    
    :param N: Description
    :type N: Optional[int]
    :param alpha: Description
    :type alpha: Optional[int]
    """
    N = N_VALUE if N is None else N
    alpha = ALPHA_0 if alpha is None else alpha

    print(f"\n[*] Question 9 : Fixed N comparison (N={N}, alpha={alpha})")

    xU = mesh_uniform(N)
    _, _, UU, eHU, eLU = FEM_1d(xU, compute_l2=True)

    xG = mesh_geometric(N, alpha)
    _, _, UG, eHG, eLG = FEM_1d(xG, compute_l2=True)

    print("\n[*] Fixed N comparison:")
    print(f"\n[*] Uniform     : err_H10={eHU:.6e}  err_L2={eLU:.6e}")
    print(f"[*] Geo a={alpha}: err_H10={eHG:.6e}  err_L2={eLG:.6e}")
        
    return eHU, xU, UU, eHG, xG, UG


def graph_u_uniform_vs_geometric (N : Optional[int] = None, alpha : Optional[int] = None) :
    """

    """
    N = N_VALUE if N is None else N
    alpha = ALPHA_0 if alpha is None else alpha

    _, xU, UU, _, xG, UG = compare_uniform_vs_geometric(N, alpha)

    plot_uniform_vs_geometric(N, alpha, xU, UU, xG, UG, u_exact, p1_eval)

    return None



# ------- Auxiliar functions --------


def _is_valid_mesh (x : Optional[List[float]] = None) -> bool :
    """
    
    """

    if len(x) <= 2 :
        return False
    
    return (np.isclose(x[0], 0.0) and np.isclose(x[1], 1.0) and np.all(np.diff(x) > 0))


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
            
            phi_L = (b - X) / h
            phi_R = (X - a) / h

            U_h_x = UL * phi_L + UR * phi_R

            diff = u_exact(X) - U_h_x

            err2 += wi * diff * diff * half

    return np.sqrt(err2)


def p1_eval (x_nodes, U_nodes, x_eval) :
    """
    Evaluate a P1 FEM function (piecewise linear) on x_eval.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    U_nodes = np.asarray(U_nodes, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    N = len(x_nodes) - 1

    idx = np.searchsorted(x_nodes, x_eval, side="right") - 1
    idx = np.clip(idx, 0, N-1)

    a = x_nodes[idx]
    b = x_nodes[idx + 1]
    h = b - a

    phi_L = (b - x_eval) / h
    phi_R = (x_eval - a) / h

    return U_nodes[idx] * phi_L + U_nodes[idx + 1] * phi_R


