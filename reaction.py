from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

from species import ChemicalSpecies


class ChemicalReaction:
    """Simple chemical reaction with kinetics and firing capability.

    A reaction is defined by its input and output stoichiometries (either
    as dictionaries keyed by species name or ``ChemicalSpecies`` objects, or
    later converted into vectors), a rate law identifier or callable, and a
    dictionary of parameters used by the rate law.

    Before rates can be computed or the reaction fired the caller must
    supply an ordered list of species via :meth:`to_vectors`.  That list is
    used both for building stoichiometric vectors and for querying
    concentrations when the rate law is evaluated.
    """

    def __init__(
        self,
        stoich_in: Dict[Union[str, ChemicalSpecies], float],
        stoich_out: Dict[Union[str, ChemicalSpecies], float],
        rate_law: Union[str, Any] = "massAction",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.stoich_in = stoich_in
        self.stoich_out = stoich_out
        self.rate_law = rate_law
        self.params = params or {}

        # these are populated by ``to_vectors``
        self.species: Optional[List[ChemicalSpecies]] = None
        self.stoich_in_vec: Optional[List[float]] = None
        self.stoich_out_vec: Optional[List[float]] = None
        self.stoich_net_vec: Optional[List[float]] = None

    def __repr__(self) -> str:
        return (
            f"ChemicalReaction(in={self.stoich_in!r}, out={self.stoich_out!r}, "
            f"law={self.rate_law!r})"
        )
  
    def calculate_rate(self) -> float:
        """Evaluate the current reaction rate using the selected law.

        The species concentrations used for evaluation are taken from the
        last entry of each ``ChemicalSpecies`` in ``self.species``.  The
        method supports a handful of built-in laws; a callable may also be
        supplied directly in ``rate_law``.  Parameter interpretation is
        documented in the tests below.
        """
        conc = self._conc_vector()
        p = self.params

        def resolve_index(key: Union[int, str, ChemicalSpecies]) -> int:
            if isinstance(key, int):
                return key
            if isinstance(key, ChemicalSpecies):
                if self.species is None:
                    raise ValueError("species order not set")
                return self.species.index(key)
            if isinstance(key, str):
                if self.species is None:
                    raise ValueError("species order not set")
                return next(i for i, s in enumerate(self.species) if s.name == key)
            raise KeyError(f"cannot resolve index for {key!r}")

        if self.rate_law == "massAction":
            k = p.get("k", 1.0)
            rate = k
            for coeff, c in zip(self.stoich_in_vec, conc):
                rate *= c ** coeff
            return rate

        elif self.rate_law == "michaelisMenten":
            Vmax = p["Vmax"]
            Km = p["Km"]
            idx = resolve_index(p["S"])
            S = conc[idx]
            return Vmax * S / (Km + S)

        elif self.rate_law == "hillA":
            K = p["K"]
            n = p["n"]
            k = p["k"]
            idx = resolve_index(p["S"])
            S = conc[idx]
            return k * (S**n) / (K**n + S**n)

        elif self.rate_law == "hillR":
            K = p["K"]
            n = p["n"]
            k = p["k"]
            idx = resolve_index(p["S"])
            S = conc[idx]
            return k * (K**n) / (K**n + S**n)

        elif callable(self.rate_law):
            return self.rate_law(conc, **p)

        else:
            raise ValueError(f"unsupported rate law: {self.rate_law}")
  
    # Builder method to turn dictionaries into vectors
    def to_vectors(self, species_list: Iterable[ChemicalSpecies]) -> None:
        """Convert stoichiometric dicts into parallel vectors.

        ``species_list`` defines the ordering used both for the vectors and
        later when rates or fluxes are computed.  The list itself is saved
        on the object so that ``fire`` and ``calculate_rate`` can locate
        concentrations.
        """
        self.species = list(species_list)
        # names used for mapping, easier to accept strings in stoichiometry
        order_names = [s.name for s in self.species]
        index_map = {name: i for i, name in enumerate(order_names)}

        def dict_to_vec(stoich: Dict[Union[str, ChemicalSpecies], float]) -> List[float]:
            vec = [0.0] * len(order_names)
            for sp, coeff in stoich.items():
                if isinstance(sp, ChemicalSpecies):
                    name = sp.name
                else:
                    name = sp
                vec[index_map[name]] = coeff
            return vec

        if isinstance(self.stoich_in, dict):
            self.stoich_in_vec = dict_to_vec(self.stoich_in)
        else:
            self.stoich_in_vec = list(self.stoich_in)

        if isinstance(self.stoich_out, dict):
            self.stoich_out_vec = dict_to_vec(self.stoich_out)
        else:
            self.stoich_out_vec = list(self.stoich_out)

        self.stoich_net_vec = [o - i for i, o in zip(self.stoich_in_vec, self.stoich_out_vec)]

    # internal helpers ------------------------------------------------------
    def _check_vectors(self) -> None:
        if self.species is None or self.stoich_net_vec is None:
            raise RuntimeError("stoichiometric vectors not initialized; call to_vectors() first")

    def _conc_vector(self) -> List[float]:
        """Return a list of current concentrations in species order."""
        if self.species is None:
            raise RuntimeError("species order not set")
        vals: List[float] = []
        for sp in self.species:
            rec = sp.get_concentration()
            vals.append(rec[-1][1] if rec else 0.0)
        return vals

    # public API ------------------------------------------------------------
    def flux_vector(self) -> List[float]:
        """Flux per species (rate * net stoichiometry)."""
        self._check_vectors()
        rate = self.calculate_rate()
        return [rate * n for n in self.stoich_net_vec]

    def fire(self, dt: float = 1.0, time: Optional[float] = None) -> None:
        """Advance concentrations by running the reaction for ``dt`` time units.

        Each species is updated at ``last_time + dt`` (or ``time`` if given)
        using its individual last concentration plus ``flux * dt``.
        """
        self._check_vectors()
        rate = self.calculate_rate()
        fluxes = self.flux_vector()
        for sp, flux in zip(self.species, fluxes):
            last_t = sp._times[-1] if sp._times else 0.0
            new_t = time if time is not None else last_t + dt
            last_c = sp.get_concentration()[-1][1] if sp.get_concentration() else 0.0
            sp.add_concentration(new_t, last_c + flux * dt)


if __name__ == "__main__":
    # create three species and give them initial values
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=2.0)
    c = ChemicalSpecies("C", initial_conc=0.0)

    rxn = ChemicalReaction({a: 1, b: 1}, {c: 1}, "massAction", {"k": 0.5})
    rxn.to_vectors([a, b, c])
    r = rxn.calculate_rate()
    assert abs(r - 1.0) < 1e-12

    # flux vector
    flux = rxn.flux_vector()
    assert flux == [ -r, -r, r ]

    # firing should consume A and B, produce C
    rxn.fire(dt=1.0)
    assert abs(a.get_concentration()[-1][1] - (1.0 - r * 1.0)) < 1e-12
    assert abs(c.get_concentration()[-1][1] - (0.0 + r * 1.0)) < 1e-12

    print("reaction.py self‑test passed")
