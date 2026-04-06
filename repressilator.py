import numpy as np
import matplotlib.pyplot as plt

from species import ChemicalSpecies
from reaction import ChemicalReaction
from network import ChemicalReactionNetwork


def build_repressilator(
    k_prod=5.0,
    k_deg=0.2,
    K=1.0,
    n=3,
):
    """
    Build a classic 3-node repressilator:
        A ─| B ─| C ─| A
    """

    # Species
    A = ChemicalSpecies("A", initial_conc=10.0, degrade=True, degradation_rate=k_deg)
    B = ChemicalSpecies("B", initial_conc=0.0, degrade=True, degradation_rate=k_deg)
    C = ChemicalSpecies("C", initial_conc=0.0, degrade=True, degradation_rate=k_deg)

    # Production reactions (repressed by previous node)
    # ∅ -> B repressed by A
    r1 = ChemicalReaction(
        stoich_in={},
        stoich_out={B: 1},
        rate_law="hillR",
        params={"k": k_prod, "K": K, "n": n, "S": A},
    )

    # ∅ -> C repressed by B
    r2 = ChemicalReaction(
        stoich_in={},
        stoich_out={C: 1},
        rate_law="hillR",
        params={"k": k_prod, "K": K, "n": n, "S": B},
    )

    # ∅ -> A repressed by C
    r3 = ChemicalReaction(
        stoich_in={},
        stoich_out={A: 1},
        rate_law="hillR",
        params={"k": k_prod, "K": K, "n": n, "S": C},
    )

    crn = ChemicalReactionNetwork(r1, r2, r3)
    return crn


def run_ode_demo(initial = {"A": 10.0, "B": 0.0, "C": 0.0}, t_span=(0, 100)):
    crn = build_repressilator()

    t_eval = np.linspace(t_span[0], t_span[1], 20 * (t_span[1] - t_span[0]))

    t, y = crn.simulate(
        initial,
        t_span,
        method="ode",
        t_eval=t_eval,
        method_solver="RK45",
    )

    A_idx = crn.get_species_index("A")
    B_idx = crn.get_species_index("B")
    C_idx = crn.get_species_index("C")

    plt.rcParams['image.interpolation'] = 'none'
    plt.figure()
    plt.plot(t, y[A_idx], label="A")
    plt.plot(t, y[B_idx], label="B")
    plt.plot(t, y[C_idx], label="C")

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Repressilator (ODE)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (t, y)


def run_ssa_demo(initial = {"A": 10.0, "B": 0.0, "C": 0.0}, t_span=(0, 100)):
    crn = build_repressilator()

    t, y = crn.simulate(
        initial,
        t_span,
        method="ssa",
        seed=42,
    )

    plt.rcParams['image.interpolation'] = 'none'
    plt.figure()
    plt.step(t, y[0], label="A", where="post")
    plt.step(t, y[1], label="B", where="post")
    plt.step(t, y[2], label="C", where="post")

    plt.xlabel("Time")
    plt.ylabel("Molecule count")
    plt.title("Repressilator (SSA)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (t, y)


def run_cle_demo(initial = {"A": 10.0, "B": 0.0, "C": 0.0}, t_span=(0, 100)):
    crn = build_repressilator()

    
    t_eval = np.linspace(t_span[0], t_span[1], 20 * (t_span[1] - t_span[0]))

    t, y = crn.simulate(
        initial,
        t_span,
        method="cle",
        t_eval=t_eval,
        seed=1,
        dt_cle=0.01,
    )

    plt.rcParams['image.interpolation'] = 'none'
    plt.figure()
    plt.plot(t, y[0], label="A")
    plt.plot(t, y[1], label="B")
    plt.plot(t, y[2], label="C")

    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Repressilator (CLE)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (t, y)


if __name__ == "__main__":
    run_ode_demo()
    run_cle_demo()
    run_ssa_demo()