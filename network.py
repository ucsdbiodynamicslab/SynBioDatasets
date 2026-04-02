"""Chemical Reaction Network (CRN) module.

This module provides classes for representing and analyzing chemical reaction
networks, including stoichiometry, rate laws, and simulation methods.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Literal
import math
from reaction import ChemicalReaction
from species import ChemicalSpecies
from scipy.integrate import solve_ivp

class ChemicalReactionNetwork:
    """Represents a complete chemical reaction network.
    
    Manages multiple reactions, computes stoichiometry matrices, flux vectors,
    and provides methods for output and simulation. Uses ChemicalReaction and
    ChemicalSpecies objects from the reaction and species modules.
    
    Attributes:
        reactions: List of ChemicalReaction objects
        species: Dict mapping species name to ChemicalSpecies object
        stoichiometry_matrix: np.ndarray of shape (n_species, n_reactions)
        _species_list: Sorted list of ChemicalSpecies objects (for consistent ordering)
        _species_index: Dict mapping species name to matrix row index
    """
    
    def __init__(self, *reactions: ChemicalReaction, 
                 species_dict: Optional[Dict[str, ChemicalSpecies]] = None):
        """Initialize a CRN with reactions and optional species definitions.
        
        Args:
            *reactions: Variable number of ChemicalReaction objects
            species_dict: Optional dict of species name -> ChemicalSpecies
        """
        if not reactions:
            raise ValueError("CRN must contain at least one reaction")
        
        self.reactions = list(reactions)
        self.species = species_dict or {}
        
        # Discover all species from reactions and extract ChemicalSpecies objects
        self._species_objects = {}  # name -> ChemicalSpecies
        for rxn in self.reactions:
            for sp in list(rxn.stoich_in.keys()) + list(rxn.stoich_out.keys()):
                if isinstance(sp, ChemicalSpecies):
                    if sp.name not in self._species_objects:
                        self._species_objects[sp.name] = sp
        
        # Add/override with species from species_dict
        for name, sp_obj in self.species.items():
            self._species_objects[name] = sp_obj
        
        # If no species were found in reactions, create defaults
        self._species_names = set(self._species_objects.keys())
        if not self._species_names:
            raise ValueError("No species found in reactions or species_dict")
        
        # Create ChemicalSpecies objects for any missing species
        for species_name in self._species_names:
            if species_name not in self._species_objects:
                self._species_objects[species_name] = ChemicalSpecies(species_name)
        
        # Update self.species with all discovered species objects
        self.species = self._species_objects
        
        # Create sorted species list and index mapping
        sorted_names = sorted(self._species_names)
        self._species_list = [self.species[name] for name in sorted_names]
        self._species_index = {name: idx for idx, name in enumerate(sorted_names)}
        
        # Convert all reactions to vectors using the species ordering
        for rxn in self.reactions:
            rxn.to_vectors(self._species_list)
        
        # Build stoichiometry matrix
        self._build_stoichiometry_matrix()
    
    def _build_stoichiometry_matrix(self):
        """Construct the stoichiometry matrix from reaction net vectors.
        
        The stoichiometry matrix S has shape (n_species, n_reactions).
        S[i,j] is the net stoichiometric coefficient for species i in reaction j.
        """
        n_species = len(self._species_list)
        n_reactions = len(self.reactions)
        self.stoichiometry_matrix = np.zeros((n_species, n_reactions), dtype=float)
        
        for rxn_idx, rxn in enumerate(self.reactions):
            if rxn.stoich_net_vec is None:
                raise RuntimeError(f"Reaction {rxn_idx} vectors not initialized")
            self.stoichiometry_matrix[:, rxn_idx] = rxn.stoich_net_vec
    
    @property
    def n_species(self) -> int:
        """Number of species in the network."""
        return len(self._species_list)
    
    @property
    def n_reactions(self) -> int:
        """Number of reactions in the network."""
        return len(self.reactions)
    
    @property
    def species_names(self) -> List[str]:
        """Sorted list of species names."""
        return list(self._species_index.keys())
    
    def get_species_index(self, species_name: str) -> int:
        """Get the index of a species in the stoichiometry matrix."""
        if species_name not in self._species_index:
            raise ValueError(f"Species '{species_name}' not found in network")
        return self._species_index[species_name]
    
    def compute_flux_vector(self, time: float = 0.0) -> np.ndarray:
        """Compute the flux vector for all reactions.
        
        Uses current concentrations from the ChemicalSpecies objects.
        
        Args:
            time: Optional time reference (not used in direct calculation)
        
        Returns:
            np.ndarray of shape (n_reactions,) with reaction rates
        """
        fluxes = np.zeros(self.n_reactions)
        for rxn_idx, rxn in enumerate(self.reactions):
            fluxes[rxn_idx] = rxn.calculate_rate()
        return fluxes
    
    def compute_species_flux(self, species_name: Optional[str] = None,
                            time: float = 0.0) -> Union[np.ndarray, float]:
        """Compute the net flux for species due to reactions and degradation.
        
        Args:
            species_name: If provided, return flux for single species. Otherwise return all.
            time: Optional time reference
        
        Returns:
            If species_name is None: np.ndarray of shape (n_species,) with net fluxes
            If species_name is provided: float with net flux for that species
        """
        reaction_fluxes = self.compute_flux_vector(time)
        
        # Compute reaction contributions to species changes
        net_fluxes = self.stoichiometry_matrix @ reaction_fluxes
        
        # Account for degradation
        for sp_idx, species_obj in enumerate(self._species_list):
            if species_obj.degrade:
                # Degradation flux: -k * [X]
                conc_record = species_obj.get_concentration()
                current_conc = conc_record[-1][1] if conc_record else 0.0
                net_fluxes[sp_idx] -= species_obj.degradation_rate * current_conc
        
        if species_name is not None:
            if species_name not in self._species_index:
                raise ValueError(f"Species '{species_name}' not found in network")
            return net_fluxes[self._species_index[species_name]]
        
        return net_fluxes
    
    def rate_law_latex(self, reaction_idx: Optional[int] = None) -> Union[str, List[str]]:
        """Generate LaTeX expressions for rate laws.
        
        Args:
            reaction_idx: If provided, return LaTeX for single reaction.
                         Otherwise return all.
        
        Returns:
            If reaction_idx is None: List of LaTeX strings
            If reaction_idx is provided: Single LaTeX string
        """
        def _rate_law_latex(rxn: ChemicalReaction, idx: int) -> str:
            """Generate LaTeX for a rate law."""
            rate_str = f"k_{{{idx}}}"
            
            if not rxn.stoich_in or not rxn.stoich_in_vec:
                return rate_str
            
            # Build product of concentrations raised to powers
            reactant_strs = []
            for sp_idx, coeff in enumerate(rxn.stoich_in_vec):
                if coeff > 0:
                    sp_name = self._species_list[sp_idx].name
                    if coeff == 1:
                        reactant_strs.append(f"[{sp_name}]")
                    else:
                        reactant_strs.append(f"[{sp_name}]^{{{int(coeff)}}}")
            
            if reactant_strs:
                return rate_str + " " + " ".join(reactant_strs)
            return rate_str
        
        if reaction_idx is not None:
            if not (0 <= reaction_idx < self.n_reactions):
                raise ValueError(f"Reaction index {reaction_idx} out of range")
            return _rate_law_latex(self.reactions[reaction_idx], reaction_idx)
        
        return [_rate_law_latex(rxn, idx) for idx, rxn in enumerate(self.reactions)]
    
    def ode_rhs(self) -> str:
        """Return the right-hand side of the ODE system as LaTeX.
        
        Returns:
            LaTeX expression showing d[X]/dt for each species
        """
        lines = []
        for sp_idx, species_obj in enumerate(self._species_list):
            species_name = species_obj.name
            flux_terms = []
            
            # Reaction contributions
            for rxn_idx, rxn in enumerate(self.reactions):
                stoich = self.stoichiometry_matrix[sp_idx, rxn_idx]
                if stoich != 0:
                    rate_latex = self.rate_law_latex(rxn_idx)
                    sign = "+" if stoich > 0 else "-"
                    coeff = abs(int(stoich)) if stoich != int(stoich) else int(stoich)
                    flux_terms.append((sign, coeff, rate_latex, stoich > 0))
            
            # Degradation contribution
            if species_obj.degrade:
                flux_terms.append(("-", 1, f"k_{{deg,{species_name}}}[{species_name}]", False))
            
            # Format the equation
            eq = f"\\frac{{d[{species_name}]}}{{dt}} = "
            if flux_terms:
                term_strs = []
                for sign, coeff, rate, _ in flux_terms:
                    if coeff > 1:
                        term_strs.append(f"{sign} {coeff} {rate}")
                    else:
                        term_strs.append(f"{sign} {rate}")
                eq += " ".join(term_strs).lstrip("+ ")
            else:
                eq += "0"
            
            lines.append(eq)
        
        return "\n".join(lines)
    
    def reactions_as_list(self) -> List[str]:
        """Return list of reaction strings in standard notation.
        
        Returns:
            List of strings like "A + B -> C + D"
        """
        result = []
        for rxn in self.reactions:
            # Build reactants string
            r_parts = []
            for sp_idx, coeff in enumerate(rxn.stoich_in_vec):
                if coeff > 0:
                    sp_name = self._species_list[sp_idx].name
                    coeff_int = int(coeff)
                    if coeff_int > 1:
                        r_parts.append(f"{coeff_int}{sp_name}")
                    else:
                        r_parts.append(sp_name)
            r_str = " + ".join(r_parts) if r_parts else "∅"
            
            # Build products string
            p_parts = []
            for sp_idx, coeff in enumerate(rxn.stoich_out_vec):
                if coeff > 0:
                    sp_name = self._species_list[sp_idx].name
                    coeff_int = int(coeff)
                    if coeff_int > 1:
                        p_parts.append(f"{coeff_int}{sp_name}")
                    else:
                        p_parts.append(sp_name)
            p_str = " + ".join(p_parts) if p_parts else "∅"
            
            result.append(f"{r_str} -> {p_str}")
        return result
    
    def network_graph_string(self, include_rates: bool = True) -> str:
        """Generate a Graphviz DOT string representing the network.
        
        Args:
            include_rates: Whether to include rate constants on edges
        
        Returns:
            String in DOT format for Graphviz
        """
        lines = ["digraph ChemicalNetwork {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=circle];')
        
        # Add species as nodes
        for species_obj in self._species_list:
            if species_obj.degrade:
                lines.append(f'  "{species_obj.name}" [label="{species_obj.name}\\n(degrade)"];')
            else:
                lines.append(f'  "{species_obj.name}";')
        
        # Add a dummy sink node for degradation
        lines.append('  "degradation" [shape=triangle, label="degrad"];')
        
        # Add reactions as nodes and edges
        for rxn_idx, rxn in enumerate(self.reactions):
            rxn_node = f"R{rxn_idx}"
            lines.append(f'  "{rxn_node}" [shape=box, label="R{rxn_idx}"];')
            
            # Reactants -> Reaction
            for sp_idx, coeff in enumerate(rxn.stoich_in_vec):
                if coeff > 0:
                    sp_name = self._species_list[sp_idx].name
                    label = f"{int(coeff)}" if coeff > 1 else ""
                    lines.append(f'  "{sp_name}" -> "{rxn_node}" [label="{label}"];')
            
            # Reaction -> Products
            for sp_idx, coeff in enumerate(rxn.stoich_out_vec):
                if coeff > 0:
                    sp_name = self._species_list[sp_idx].name
                    label = f"{int(coeff)}" if coeff > 1 else ""
                    lines.append(f'  "{rxn_node}" -> "{sp_name}" [label="{label}"];')
        
        # Add degradation edges
        for species_obj in self._species_list:
            if species_obj.degrade:
                label = f"k={species_obj.degradation_rate:.2e}" if species_obj.degradation_rate > 0 else ""
                lines.append(f'  "{species_obj.name}" -> "degradation" [label="{label}"];')
        
        lines.append("}")
        return "\n".join(lines)
    
    def __repr__(self):
        return (f"ChemicalReactionNetwork(n_species={self.n_species}, "
                f"n_reactions={self.n_reactions})")
    
    # ========== Simulation methods ==========
    
    def simulate(self, initial_concentrations: Dict[str, float],
                 t_span: Tuple[float, float],
                 method: Literal["ode", "ssa", "cle"] = "ode",
                 t_eval: Optional[np.ndarray] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the CRN using the selected method.
        
        Updates the ChemicalSpecies objects with time-series concentration data
        and returns the solution.
        
        Args:
            initial_concentrations: Dict mapping species names to initial conc.
            t_span: Tuple of (t_start, t_end)
            method: One of "ode", "ssa", or "cle"
            t_eval: Optional array of times at which to compute solution (for ODE/CLE)
            **kwargs: Method-specific options
                - For ODE: dense_output, events, etc. (scipy.integrate.solve_ivp args)
                - For SSA: seed (int) for random number generation
                - For CLE: seed for random number generation
        
        Returns:
            Tuple of (t, y) where t are times and y has shape (n_species, len(t))
        """
        # Initialize species with given concentrations
        t0 = t_span[0]
        for species_name, conc in initial_concentrations.items():
            if species_name in self.species:
                self.species[species_name]._times = [t0]
                self.species[species_name]._concentrations = [conc]
        
        if method == "ode":
            return self.simulate_ode(initial_concentrations, t_span, t_eval, **kwargs)
        elif method == "ssa":
            return self.simulate_ssa(initial_concentrations, t_span[1], **kwargs)
        elif method == "cle":
            return self.simulate_cle(initial_concentrations, t_span, t_eval, **kwargs)
        else:
            raise ValueError(f"Unknown simulation method: {method}. "
                           f"Choose from 'ode', 'ssa', or 'cle'")
    
    def simulate_ode(self, initial_concentrations: Dict[str, float],
                     t_span: Tuple[float, float],
                     t_eval: Optional[np.ndarray] = None,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the CRN using ODE solvers (scipy.integrate.solve_ivp).
        
        Args:
            initial_concentrations: Dict mapping species names to initial conc.
            t_span: Tuple of (t_start, t_end)
            t_eval: Optional array of times at which to compute solution
            **kwargs: Additional arguments to pass to solve_ivp (method, dense_output, etc.)
        
        Returns:
            Tuple of (t, y) where y has shape (n_species, len(t))
        """
        # Initial condition vector
        y0 = np.zeros(self.n_species)
        for name, conc in initial_concentrations.items():
            if name in self._species_index:
                y0[self._species_index[name]] = conc
        
        # Define RHS function: dy/dt = S @ v(y)
        def rhs(t, y):
            # Update species concentrations temporarily for rate calculations
            for sp_idx, sp in enumerate(self._species_list):
                if sp._times and sp._concentrations:
                    # Store last value temporarily
                    old_times = sp._times[:]
                    old_concs = sp._concentrations[:]
                else:
                    old_times = []
                    old_concs = []
                
                # Set concentration at current time
                sp._times = [t]
                sp._concentrations = [y[sp_idx]]
            
            # Calculate fluxes
            flux = self.compute_species_flux(time=t)
            
            # Restore original times/concentrations (not used further in integration)
            # Actually we don't need to restore since solve_ivp manages the state
            
            return flux
        
        # Solve ODE
        sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, **kwargs)
        
        # Update species objects with the solution
        for sp_idx, sp in enumerate(self._species_list):
            sp._times = list(sol.t)
            sp._concentrations = list(sol.y[sp_idx])
        
        return sol.t, sol.y
    
    def simulate_ssa(self, initial_concentrations: Dict[str, float],
                     t_end: float,
                     seed: Optional[int] = None,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the CRN using the Stochastic Simulation Algorithm (Gillespie).
        
        Args:
            initial_concentrations: Dict mapping species names to initial counts (integers)
            t_end: End time for simulation
            seed: Optional random seed for reproducibility
            **kwargs: Additional options (currently unused)
        
        Returns:
            Tuple of (times, trajectory) where trajectory has shape (n_species, n_events)
        """
        if seed is not None:
            np.random.seed(seed)
        
        times = [0.0]
        trajectory = [np.array([initial_concentrations.get(name, 0.0) 
                               for name in self.species_names])]
        
        t = 0.0
        
        # Initialize species with integer counts
        for name, conc in initial_concentrations.items():
            if name in self.species:
                self.species[name]._times = [t]
                self.species[name]._concentrations = [float(conc)]
        
        while t < t_end:
            # Calculate propensities for all reactions
            propensities = self.compute_flux_vector(time=t)
            
            # Total propensity
            a0 = np.sum(propensities)
            
            if a0 <= 0:
                # No more reactions can fire - simulation stalled
                break
            
            # Sample time to next reaction: tau ~ Exp(a0)
            tau = np.random.exponential(1.0 / a0)
            t_next = t + tau
            
            # If we've exceeded t_end, stop
            if t_next > t_end:
                # Record the final state at t_end with degradation applied
                final_state = np.array([sp.get_concentration()[-1][1] 
                                       for sp in self._species_list])
                times.append(t_end)
                trajectory.append(final_state)
                break
            
            # Choose reaction according to propensities
            probabilities = propensities / a0
            reaction_idx = np.random.choice(len(self.reactions), p=probabilities)
            
            # Fire the chosen reaction
            rxn = self.reactions[reaction_idx]
            for sp, net in zip(self._species_list, rxn.stoich_net_vec):
                last_c = sp.get_concentration()[-1][1]
                sp.add_concentration(t_next, last_c + net)   # net is exactly -1 or +1
            
            # Apply degradation to all species
            for sp in self._species_list:
                if sp.degrade:
                    # Advance degradation: c(t+tau) = c(t) * exp(-k*tau)
                    conc_record = sp.get_concentration()
                    current_conc = conc_record[-1][1] if conc_record else 0.0
                    new_conc = current_conc * math.exp(-sp.degradation_rate * tau)
                    sp.add_concentration(t_next, new_conc)
            
            t = t_next
            times.append(t)
            final_state = np.array([sp.get_concentration()[-1][1] 
                                   for sp in self._species_list])
            trajectory.append(final_state)
        
        return np.array(times), np.array(trajectory).T
    
    def simulate_cle(self, initial_concentrations: Dict[str, float],
                     t_span: Tuple[float, float],
                     t_eval: Optional[np.ndarray] = None,
                     seed: Optional[int] = None,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the CRN using the Chemical Langevin Equation.
        
        The CLE is an intermediate between deterministic ODEs and the SSA,
        valid in the regime of moderate copy numbers. Includes stochastic
        noise terms but is deterministic in reactions.
        
        Args:
            initial_concentrations: Dict mapping species names to initial conc.
            t_span: Tuple of (t_start, t_end)
            t_eval: Optional array of times at which to compute solution
            seed: Optional random seed for reproducibility
            **kwargs: Additional arguments for numerical solution
        
        Returns:
            Tuple of (t, y) where y has shape (n_species, len(t))
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initial condition vector
        y0 = np.zeros(self.n_species)
        for name, conc in initial_concentrations.items():
            if name in self._species_index:
                y0[self._species_index[name]] = conc
        
        # Stoichiometry matrix (for noise term)
        S = self.stoichiometry_matrix
        
        t0, tf = t_span
        if t_eval is None:
            t_eval = np.linspace(t0, tf, 100)
        else:
            t_eval = np.asarray(t_eval)
        
        # Manual integration with stochastic noise (Euler-Maruyama)
        dt_base = kwargs.get('dt_cle', (tf - t0) / min(1000, 10 * len(t_eval)))
        
        times = []
        trajectory = []
        
        y = y0.copy()
        t = t0
        eval_idx = 0
        
        # Record initial state
        times.append(t)
        trajectory.append(y.copy())
        eval_idx += 1
        
        while t < tf - 1e-10 and eval_idx < len(t_eval):
            # Next evaluation time
            t_next = t_eval[eval_idx]
            
            # Take steps toward t_next
            while t < t_next - 1e-10:
                dt = min(dt_base, t_next - t)
                
                # Clamp to non-negative
                y = np.maximum(y, 0.0)
                
                # Update species for rate calculations
                for sp_idx in range(self.n_species):
                    self._species_list[sp_idx]._times = [t]
                    self._species_list[sp_idx]._concentrations = [y[sp_idx]]
                
                # Calculate propensities (reaction rates only)
                v = self.compute_flux_vector(time=t)
                
                # Drift: reaction contributions
                drift = S @ v
                
                # Add degradation drift
                for sp_idx, sp in enumerate(self._species_list):
                    if sp.degrade:
                        drift[sp_idx] -= sp.degradation_rate * y[sp_idx]
                
                # Diffusion: stochastic noise (only from reactions)
                dW = np.random.randn(self.n_reactions)
                noise = np.sqrt(np.maximum(v, 0.0))
                diffusion = S @ (noise * dW)
                
                # Euler-Maruyama update
                y = y + drift * dt + diffusion * np.sqrt(dt)
                y = np.maximum(y, 0.0)
                
                t = t + dt
            
            # Record at evaluation point
            times.append(t_eval[eval_idx])
            trajectory.append(y.copy())
            eval_idx += 1
        
        # Ensure we have the final time if needed
        if len(times) < len(t_eval):
            times.append(tf)
            trajectory.append(y.copy())
        
        # Update species objects with the solution
        times_arr = np.array(times[:len(t_eval)])
        traj_arr = np.array(trajectory[:len(t_eval)]).T
        
        for sp_idx, sp in enumerate(self._species_list):
            sp._times = list(times_arr)
            sp._concentrations = list(traj_arr[sp_idx])
        
        return times_arr, traj_arr


# ============================================================================
# Self-tests
# ============================================================================

def test_crn_creation():
    """Test CRN creation with multiple ChemicalReaction objects."""
    # Create species
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=2.0)
    c = ChemicalSpecies("C", initial_conc=0.0)
    d = ChemicalSpecies("D", initial_conc=0.0)
    
    # Create reactions
    rxn1 = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    rxn2 = ChemicalReaction({b: 1}, {c: 1}, "massAction", {"k": 0.3})
    rxn3 = ChemicalReaction({a: 1, b: 1}, {d: 1}, "massAction", {"k": 0.1})
    
    crn = ChemicalReactionNetwork(rxn1, rxn2, rxn3)
    
    assert crn.n_species == 4, f"Expected 4 species, got {crn.n_species}"
    assert crn.n_reactions == 3, f"Expected 3 reactions, got {crn.n_reactions}"
    assert set(crn.species_names) == {"A", "B", "C", "D"}
    
    print("✓ test_crn_creation passed")


def test_stoichiometry_matrix():
    """Test stoichiometry matrix construction."""
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=2.0)
    c = ChemicalSpecies("C", initial_conc=0.0)
    
    rxn1 = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    rxn2 = ChemicalReaction({b: 1}, {c: 1}, "massAction", {"k": 0.3})
    
    crn = ChemicalReactionNetwork(rxn1, rxn2)
    S = crn.stoichiometry_matrix
    
    # Check shape
    assert S.shape == (3, 2)
    
    # Check values for rxn1: A -> B
    # A should have -1, B should have +1
    assert S[crn.get_species_index("A"), 0] == -1
    assert S[crn.get_species_index("B"), 0] == 1
    
    # Check values for rxn2: B -> C
    # B should have -1, C should have +1
    assert S[crn.get_species_index("B"), 1] == -1
    assert S[crn.get_species_index("C"), 1] == 1
    
    print("✓ test_stoichiometry_matrix passed")


def test_flux_computation():
    """Test reaction and species flux computation."""
    # Simple system: A -> B -> C
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=2.0)
    c = ChemicalSpecies("C", initial_conc=0.0)
    
    rxn1 = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    rxn2 = ChemicalReaction({b: 1}, {c: 1}, "massAction", {"k": 0.3})
    
    crn = ChemicalReactionNetwork(rxn1, rxn2)
    
    reaction_fluxes = crn.compute_flux_vector()
    
    assert reaction_fluxes.shape == (2,)
    assert np.isclose(reaction_fluxes[0], 0.5 * 1.0), f"Expected 0.5, got {reaction_fluxes[0]}"
    assert np.isclose(reaction_fluxes[1], 0.3 * 2.0), f"Expected 0.6, got {reaction_fluxes[1]}"
    
    # Test species flux
    species_flux = crn.compute_species_flux()
    assert species_flux.shape == (3,)
    
    # A: -0.5 * 1.0 = -0.5
    assert np.isclose(species_flux[crn.get_species_index("A")], -0.5)
    
    # B: +0.5 * 1.0 - 0.3 * 2.0 = 0.5 - 0.6 = -0.1
    assert np.isclose(species_flux[crn.get_species_index("B")], -0.1)
    
    # C: +0.3 * 2.0 = 0.6
    assert np.isclose(species_flux[crn.get_species_index("C")], 0.6)
    
    print("✓ test_flux_computation passed")


def test_flux_with_degradation():
    """Test flux computation including degradation terms."""
    # Create species with degradation
    a = ChemicalSpecies("A", initial_conc=2.0, degrade=True, degradation_rate=0.1)
    b = ChemicalSpecies("B", initial_conc=1.0, degrade=False)
    
    rxn = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    crn = ChemicalReactionNetwork(rxn, species_dict={"A": a, "B": b})
    
    species_flux = crn.compute_species_flux()
    
    # A: -0.5 * 2.0 (reaction) - 0.1 * 2.0 (degradation) = -1.2
    assert np.isclose(species_flux[crn.get_species_index("A")], -1.2), \
        f"Expected -1.2, got {species_flux[crn.get_species_index('A')]}"
    
    # B: +0.5 * 2.0 = 1.0
    assert np.isclose(species_flux[crn.get_species_index("B")], 1.0)
    
    print("✓ test_flux_with_degradation passed")


def test_reaction_list_output():
    """Test reaction output as list."""
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    c = ChemicalSpecies("C", initial_conc=0.0)
    d = ChemicalSpecies("D", initial_conc=0.0)
    
    rxn1 = ChemicalReaction({a: 1, b: 1}, {c: 2}, "massAction", {"k": 0.5})
    rxn2 = ChemicalReaction({c: 1}, {a: 1, d: 1}, "massAction", {"k": 0.3})
    
    crn = ChemicalReactionNetwork(rxn1, rxn2)
    rxn_list = crn.reactions_as_list()
    
    assert len(rxn_list) == 2
    
    print("✓ test_reaction_list_output passed")


def test_graph_output():
    """Test graph string generation."""
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    c = ChemicalSpecies("C", initial_conc=0.0)
    
    rxn1 = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    rxn2 = ChemicalReaction({b: 1}, {c: 1}, "massAction", {"k": 0.3})
    
    crn = ChemicalReactionNetwork(rxn1, rxn2)
    graph_str = crn.network_graph_string()
    
    # Check it's valid DOT format
    assert graph_str.startswith("digraph")
    assert "}" in graph_str
    assert "A" in graph_str
    assert "B" in graph_str
    assert "C" in graph_str
    
    print("✓ test_graph_output passed")


def test_rate_law_latex():
    """Test LaTeX rate law generation."""
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    c = ChemicalSpecies("C", initial_conc=0.0)
    
    rxn1 = ChemicalReaction({a: 1, b: 2}, {c: 1}, "massAction", {"k": 0.5})
    rxn2 = ChemicalReaction({c: 1}, {a: 1}, "massAction", {"k": 0.3})
    
    crn = ChemicalReactionNetwork(rxn1, rxn2)
    
    # Test single reaction
    rate_latex = crn.rate_law_latex(0)
    assert "[A]" in rate_latex or "A" in rate_latex, f"LaTeX should contain A: {rate_latex}"
    
    # Test all reactions
    all_rates = crn.rate_law_latex()
    assert len(all_rates) == 2
    
    print("✓ test_rate_law_latex passed")


def test_ode_rhs_output():
    """Test ODE RHS generation."""
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    c = ChemicalSpecies("C", initial_conc=0.0)
    
    rxn1 = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    rxn2 = ChemicalReaction({b: 1}, {c: 1}, "massAction", {"k": 0.3})
    
    crn = ChemicalReactionNetwork(rxn1, rxn2)
    ode_str = crn.ode_rhs()
    
    assert "d[A]" in ode_str
    assert "d[B]" in ode_str
    assert "d[C]" in ode_str
    
    print("✓ test_ode_rhs_output passed")


def test_ode_simulation():
    """Test ODE simulation method."""
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    
    rxn = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    crn = ChemicalReactionNetwork(rxn)
    
    # Simulate from t=0 to t=1
    t, y = crn.simulate_ode({"A": 1.0, "B": 0.0}, (0, 1), t_eval=np.linspace(0, 1, 11))
    
    assert t.shape[0] > 0, "Should have time points"
    assert y.shape == (2, t.shape[0]), f"Expected shape (2, {t.shape[0]}), got {y.shape}"
    assert np.isclose(y[0, 0], 1.0), "Initial A concentration should be 1.0"
    assert np.isclose(y[1, 0], 0.0), "Initial B concentration should be 0.0"
    assert y[0, -1] < 1.0, "A should decrease over time"
    assert y[1, -1] > 0.0, "B should increase over time"
    
    # Check conservation: A + B should be conserved
    total = y[0, :] + y[1, :]
    assert np.allclose(total, 1.0), "A + B should be conserved (= 1.0)"
    
    print("✓ test_ode_simulation passed")


def test_ssa_simulation():
    """Test SSA (Gillespie) simulation method."""
    a = ChemicalSpecies("A", initial_conc=100)
    b = ChemicalSpecies("B", initial_conc=0)
    
    rxn = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 1.0})
    crn = ChemicalReactionNetwork(rxn)
    
    # Simulate from t=0 to t=5
    t, y = crn.simulate_ssa({"A": 100, "B": 0}, 5.0, seed=42)
    
    assert t.shape[0] > 0, "Should have time points"
    assert y.shape == (2, t.shape[0]), f"Expected shape (2, {t.shape[0]}), got {y.shape}"
    assert np.isclose(t[0], 0.0), "First time should be 0"
    assert t[-1] <= 5.0, "Last time should be <= t_end"
    assert np.isclose(y[0, 0], 100), "Initial A should be 100"
    assert np.isclose(y[1, 0], 0), "Initial B should be 0"
    assert y[0, -1] <= 100, "A should not increase"
    assert y[1, -1] >= 0, "B should not decrease"
    
    # Check conservation
    total = y[0, :] + y[1, :]
    assert np.allclose(total, 100), "A + B should be conserved (= 100)"
    
    print("✓ test_ssa_simulation passed")


def test_cle_simulation():
    """Test CLE simulation method."""
    # Use larger rate for more obvious decrease
    a = ChemicalSpecies("A", initial_conc=10.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    
    rxn = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 2.0})
    crn = ChemicalReactionNetwork(rxn)
    
    # Simulate from t=0 to t=2
    t, y = crn.simulate_cle({"A": 10.0, "B": 0.0}, (0, 2), 
                            t_eval=np.linspace(0, 2, 21), seed=42, dt_cle=0.01)
    
    assert t.shape[0] > 0, "Should have time points"
    assert y.shape == (2, t.shape[0]), f"Expected shape (2, {t.shape[0]}), got {y.shape}"
    assert np.isclose(y[0, 0], 10.0), "Initial A concentration should be 10.0"
    assert np.isclose(y[1, 0], 0.0, atol=1e-10), "Initial B concentration should be 0.0"
    
    # Over the long term, A should decrease (on average) and B should increase (on average)
    # With stochastic noise, check the mean tendency
    A_vals = y[0, :]
    B_vals = y[1, :]
    
    # A should be less at final time than at start
    assert A_vals[-1] < A_vals[0], f"A should tend to decrease: {A_vals[0]:.2f} -> {A_vals[-1]:.2f}"
    
    # B should be more at final time than at start
    assert B_vals[-1] > B_vals[0], f"B should tend to increase: {B_vals[0]:.2f} -> {B_vals[-1]:.2f}"
    
    # Check all concentrations are non-negative
    assert np.all(y >= -1e-10), "Concentrations should be non-negative"
    
    print("✓ test_cle_simulation passed")


def test_simulate_wrapper():
    """Test the main simulate method with all three approaches."""
    a = ChemicalSpecies("A", initial_conc=1.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    
    rxn = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    crn = ChemicalReactionNetwork(rxn)
    
    # Test ODE method
    t_ode, y_ode = crn.simulate({"A": 1.0, "B": 0.0}, (0, 1), method="ode",
                                t_eval=np.linspace(0, 1, 11))
    assert y_ode.shape[0] == 2
    
    # Reset species for next simulation
    a._times = [0.0]
    a._concentrations = [1.0]
    b._times = [0.0]
    b._concentrations = [0.0]
    
    # Test SSA method
    t_ssa, y_ssa = crn.simulate({"A": 50, "B": 0}, (0, 2), method="ssa", seed=42)
    assert y_ssa.shape[0] == 2
    
    # Reset species
    a._times = [0.0]
    a._concentrations = [1.0]
    b._times = [0.0]
    b._concentrations = [0.0]
    
    # Test CLE method
    t_cle, y_cle = crn.simulate({"A": 1.0, "B": 0.0}, (0, 1), method="cle",
                                t_eval=np.linspace(0, 1, 11), seed=42)
    assert y_cle.shape[0] == 2
    
    # Test invalid method
    try:
        crn.simulate({"A": 1.0, "B": 0.0}, (0, 1), method="invalid")
        assert False, "Should raise ValueError for invalid method"
    except ValueError as e:
        assert "Unknown simulation method" in str(e)
    
    print("✓ test_simulate_wrapper passed")


def test_simulation_with_degradation():
    """Test simulation with degradation."""
    a = ChemicalSpecies("A", initial_conc=1.0, degrade=True, degradation_rate=1.0)
    b = ChemicalSpecies("B", initial_conc=0.0)
    
    # A decays to B, and A also degrades
    rxn = ChemicalReaction({a: 1}, {b: 1}, "massAction", {"k": 0.5})
    crn = ChemicalReactionNetwork(rxn, species_dict={"A": a, "B": b})
    
    t, y = crn.simulate_ode({"A": 1.0, "B": 0.0}, (0, 2), t_eval=np.linspace(0, 2, 21))
    
    # A should decay faster due to both reaction and degradation
    assert y[0, -1] < 0.1, "A should decay significantly due to reaction + degradation"
    
    # Total should decrease due to degradation
    total_initial = y[0, 0] + y[1, 0]
    total_final = y[0, -1] + y[1, -1]
    assert total_final < total_initial, "Total concentration should decrease due to degradation"
    
    print("✓ test_simulation_with_degradation passed")


if __name__ == "__main__":
    # Run all self-tests
    test_crn_creation()
    test_stoichiometry_matrix()
    test_flux_computation()
    test_flux_with_degradation()
    test_reaction_list_output()
    test_graph_output()
    test_rate_law_latex()
    test_ode_rhs_output()
    test_ode_simulation()
    test_ssa_simulation()
    test_cle_simulation()
    test_simulate_wrapper()
    test_simulation_with_degradation()
    
    print("\n" + "="*50)
    print("All self-tests passed!")
    print("="*50)
