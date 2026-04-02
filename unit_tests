import unittest
import numpy as np
from species import ChemicalSpecies
from reaction import ChemicalReaction
from network import ChemicalReactionNetwork

class TestCRNSimulation(unittest.TestCase):

    def setUp(self):
        """Set up a simple A -> B reaction network for testing."""
        self.a = ChemicalSpecies("A", initial_conc=100.0)
        self.b = ChemicalSpecies("B", initial_conc=0.0)
        self.rxn = ChemicalReaction({self.a: 1}, {self.b: 1}, "massAction", {"k": 0.5})
        self.crn = ChemicalReactionNetwork(self.rxn)
        self.initial_state = {"A": 100.0, "B": 0.0}
        self.t_span = (0, 5)

    def test_ode_deterministic_flow(self):
        """Test if ODE simulation reduces reactant and increases product."""
        t, y = self.crn.simulate(self.initial_state, self.t_span, method="ode")
        self.assertLess(y[0, -1], 100.0)
        self.assertGreater(y[1, -1], 0.0)
        total_mass = y[0, -1] + y[1, -1]
        self.assertAlmostEqual(total_mass, 100.0, places=5)

    def test_ssa_stochastic_integers(self):
        """Test if SSA returns discrete steps and honors the seed."""
        # Using a short span to check for stochastic changes
        t1, y1 = self.crn.simulate(self.initial_state, (0, 1), method="ssa", seed=42)
        t2, y2 = self.crn.simulate(self.initial_state, (0, 1), method="ssa", seed=42)
        
        # Test reproducibility with seed
        np.testing.assert_array_equal(y1, y2)
        
        # Verify that changes are integers (since net is +/- 1)
        diffs = np.diff(y1[0, :])
        for d in diffs:
            if d != 0:
                self.assertEqual(float(d), float(int(d)))

    def test_cle_noise_reproducibility(self):
        """Test if CLE is reproducible with a seed."""
        t_eval = np.linspace(0, 5, 50)
        t1, y1 = self.crn.simulate_cle(self.initial_state, self.t_span, t_eval=t_eval, seed=123)
        t2, y2 = self.crn.simulate_cle(self.initial_state, self.t_span, t_eval=t_eval, seed=123)
        np.testing.assert_array_almost_equal(y1, y2)

    def test_degradation_handling(self):
        """Test if species degradation is correctly applied in ODE."""
        # We create a species with a high degradation rate
        prot = ChemicalSpecies("Protein", initial_conc=10.0, degrade=True, degradation_rate=1.0)
        
        # FIX: We must provide at least one reaction to satisfy the CRN constructor.
        # We create a 'dummy' reaction that does nothing (k=0)
        dummy_rxn = ChemicalReaction({prot: 1}, {prot: 1}, "massAction", {"k": 0.0})
        crn_deg = ChemicalReactionNetwork(dummy_rxn)
        
        t, y = crn_deg.simulate({"Protein": 10.0}, (0, 2), method="ode", t_eval=np.linspace(0, 2, 10))
        
        # With k=1.0 and t=2, concentration should drop significantly
        # Expected: 10 * e^(-2) approx 1.35
        self.assertLess(y[0, -1], 2.0)
        self.assertGreater(y[0, -1], 1.0)

    def test_wrapper_dispatch(self):
        """Ensure the simulate() wrapper correctly dispatches to specific methods."""
        t, y = self.crn.simulate(self.initial_state, self.t_span, method="ode")
        self.assertIsInstance(y, np.ndarray)

    def test_species_inplace_update(self):
        """Test if species objects are updated after simulation."""
        self.crn.simulate(self.initial_state, self.t_span, method="ode")
        spec_a = self.crn.species["A"]
        self.assertTrue(len(spec_a._concentrations) > 0)

if __name__ == '__main__':
    unittest.main()
