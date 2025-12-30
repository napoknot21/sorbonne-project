from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


def plot_convergence (N_values, err, title : str, ylabel) :
    """
    
    """
    plt.figure()

    plt.loglog(N_values, err, marker="o")
    plt.grid(True, which="both")

    plt.xlabel("N (number of elements)")
    plt.ylabel(ylabel)

    plt.title(title)

    return None


def plot_meshes (N, alphas, func) :
    """
    Docstring for plot_meshes
    
    :param N: Description
    :type N: int
    :param alphas: Description
    :type alphas: Tuple
    """

    plt.figure()

    for a in alphas :

        x = func(N, a)
        plt.plot(x, 0*x + a, marker=".", linestyle="none", label=f"alpha={a}")

    plt.grid(True, which="both")
    plt.xlabel("x_i")
    plt.ylabel("Alpha (just to separate line)")
    
    plt.title(f"Geometric meshes (N={N})")
    plt.legend()

    return None


def plot_alphas (N, alphas, err_H, err_L2, label_H, label_L2, x_label = "alpha") :

    plt.figure()
    
    plt.semilogy(alphas, err_H, marker="O", label=label_H)
    plt.semilogy(alphas, err_L2, marker="O", label=label_L2)

    plt.grid(True, which="both")

    plt.xlabel(x_label)
    plt.ylabel(f"Error (N={N})")

    plt.title(f"Error vs alpha (fixed N={N})")

    plt.legend() 

    return None


def plot_uniform_vs_geometric (N, alpha, xU, UU, xG, UG, u_exact, p1_eval) :
    """
    Docstring for plot_uniform_vs_geometric
    """

    Xfine = np.linspace(0.0, 1.0, 2000)

    plt.figure()
    
    plt.plot(Xfine, u_exact(Xfine), label="u exact")
    plt.plot(Xfine, p1_eval(xU, UU, Xfine), label=f"u_h uniform (N={N})")
    plt.plot(Xfine, p1_eval(xG, UG, Xfine), label=f"u_h geometric (N={N}, alpha={alpha})")
    
    plt.grid(True)
    
    plt.xlabel("x")
    plt.ylabel("u")
    
    plt.title(f"Solutions (N={N}): uniform vs geometric")
    
    plt.legend()
    #plt.show()

    return None