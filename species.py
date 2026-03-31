import math


class ChemicalSpecies:
    """Simple representation of a chemical species with time-varying concentration.

    Attributes:
        name (str): unique identifier of the species.
        degrade (bool): whether the species is subject to degradation.
        degradation_rate (float): first-order rate constant (k) used when
            ``degrade`` is ``True``. Units are assumed to be consistent with
            the time values stored with concentrations.

    The concentration history is stored natively as a time series: two
    parallel lists, ``_times`` and ``_concentrations``.  The public API
    exposes helper methods to append new measurements and to retrieve either
    the full series or a value at a particular time.
    """

    def __init__(self,
                 name: str,
                 initial_conc: float = 0.0,
                 degrade: bool = False,
                 degradation_rate: float = 0.0):
        self.name = name
        self.degrade = degrade
        self.degradation_rate = degradation_rate

        # time series storage; ``times`` in the same units as rate constant
        self._times = []  # list of float
        self._concentrations = []  # list of float

        # optionally seed with an initial value at t=0
        if initial_conc is not None:
            self._times.append(0.0)
            self._concentrations.append(float(initial_conc))

    # --- concentration time-series helpers ---------------------------------
    def add_concentration(self, time: float, concentration: float):
        """Record a concentration value at a given time.

        The ``time`` may be non‑monotonic; values are stored in the order they
        are added.  A simple lookup helper uses the most recent point
        preceding a query.
        """
        self._times.append(float(time))
        self._concentrations.append(float(concentration))

    def get_concentration(self, time: float | None = None):
        """Return concentrations.

        If ``time`` is ``None`` the full list of ``(time, concentration)``
        tuples is returned.  If ``time`` is provided, the most recent
        measurement taken at or before that time is returned (``None`` if no
        such point exists).
        """
        if time is None:
            return list(zip(self._times, self._concentrations))
        # find last measurement at or before ``time``
        for t, c in reversed(list(zip(self._times, self._concentrations))):
            if t <= time:
                return c
        return None

    # --- degradation support -----------------------------------------------
    def apply_degradation(self, dt: float):
        """Advance the species by ``dt`` units of time, applying degradation.

        A simple first-order decay is used: ``c(t+dt) = c(t) * exp(-k * dt)``.
        The new concentration is appended to the time series.  If ``degrade``
        is ``False`` the method is a no-op.
        """
        if not self.degrade:
            return
        if not self._concentrations:
            # nothing to degrade
            return

        last_time = self._times[-1]
        last_conc = self._concentrations[-1]
        new_conc = last_conc * math.exp(-self.degradation_rate * dt)
        self.add_concentration(last_time + dt, new_conc)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return (
            f"ChemicalSpecies(name={self.name!r}, "
            f"current_conc={self._concentrations[-1] if self._concentrations else None}, "
            f"degrade={self.degrade}, rate={self.degradation_rate})"
        )


if __name__ == "__main__":
    # quick sanity checks
    s = ChemicalSpecies("X", initial_conc=5.0)
    assert s.get_concentration() == [(0.0, 5.0)]

    s.add_concentration(1.0, 4.0)
    assert s.get_concentration(0.5) == 5.0
    assert s.get_concentration(1.0) == 4.0

    s2 = ChemicalSpecies("Y", initial_conc=10, degrade=True, degradation_rate=1.0)
    s2.apply_degradation(0.0)
    assert abs(s2.get_concentration(0) - 10) < 1e-12
    s2.apply_degradation(1.0)
    assert abs(s2.get_concentration(1.0) - 10 * math.exp(-1)) < 1e-12

    print("species.py self-test passed")
