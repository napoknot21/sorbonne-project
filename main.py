import os
import matplotlib.pyplot as plt

from src.lib import *
from src.params import N_VALUES, N_VALUE, ALPHAS, ALPHA_0


FIG_DIR = "figures"
DPI = 300


def _ensure_dir() :
    os.makedirs(FIG_DIR, exist_ok=True)


def _save_new_figs (fig_nums_before, filenames):
    """
    Save all newly created figures since fig_nums_before as PNG.
    """
    _ensure_dir()
    fig_nums_after = set(plt.get_fignums())
    new_nums = sorted(list(fig_nums_after - set(fig_nums_before)))

    if len(new_nums) != len(filenames) :
        print(
            f"[!] Warning: expected {len(filenames)} figures but found {len(new_nums)}"
        )

    for i, num in enumerate(new_nums) :

        name = filenames[min(i, len(filenames) - 1)]
        path = os.path.join(FIG_DIR, name)
        
        plt.figure(num)
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        
        print("\n[*] Saved:", path)


def main() :

    plt.close("all")

    # ---------------- Q5: uniform convergence (2 figures) ----------------
    
    before = plt.get_fignums()
    graph_n_estimation_uniform(N_VALUES)
    _save_new_figs(
        before,
        ["q5_uniform_energy.png", "q5_uniform_l2.png"],
    )

    print("\n===========================================")

    # ---------------- Q6: meshes (1 figure) ----------------
    
    before = plt.get_fignums()
    mesh_trace(N_VALUE, ALPHAS)
    _save_new_figs(
        before,
        ["q6_meshes.png"],
    )

    print("\n===========================================")

    # ---------------- Q7: geometric convergence alpha0 (2 figures) ----------------
    
    before = plt.get_fignums()
    graph_n_estimation_geometric(N_VALUES, ALPHA_0)
    _save_new_figs(
        before,
        ["q7_geo_energy.png", "q7_geo_l2.png"],
    )

    print("\n===========================================")

    # ---------------- Q8: error vs alpha (1 figure) ----------------
    before = plt.get_fignums()
    graph_alpha_comparaison(N_VALUE, ALPHAS)
    _save_new_figs(
        before,
        ["q8_error_vs_alpha.png"],
    )

    print("\n===========================================")

    # ---------------- Q9: solutions (1 figure) ----------------
    before = plt.get_fignums()
    graph_u_uniform_vs_geometric(N_VALUE, ALPHA_0)
    _save_new_figs(
        before,
        ["q9_solutions.png"],
    )

    plt.close("all")
    print("\n[*] All PNG figures generated in ./figures/")


if __name__ == "__main__" :
    main()