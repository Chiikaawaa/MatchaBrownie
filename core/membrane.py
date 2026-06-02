import numpy as np


class Membrane:
    """
    A planar membrane perpendicular to one axis at a given position.

    Crossing probability is derived from the physical apparent permeability
    (Papp, m/s) using the formula:

        P_cross = Papp × sqrt(π × dt / D_eff)

    This ensures the macroscopic flux is invariant to the simulation timestep:
    halving dt halves P_cross per step but doubles attempt frequency, keeping
    total crossing rate constant. A raw static probability fails this invariance.

    Parameters
    ----------
    axis : int
        0=x, 1=y, 2=z — axis the membrane is perpendicular to
    position : float
        Position along that axis in metres
    Papp_ms : float
        Apparent permeability in m/s (SI).
        Convert from cm/s: multiply by 1e-2.
        Typical values:
            BBB tight junction    ~1e-7 m/s
            Intestinal epithelium ~1e-5 m/s
            Skin stratum corneum  ~1e-8 m/s
            Lung alveolar         ~1e-4 m/s
    D_eff : float
        Effective diffusion coefficient of the drug in this tissue (m²/s).
        Used to non-dimensionalise the permeability.
    dt : float
        Simulation timestep in seconds.
    """

    def __init__(
        self,
        axis:     int,
        position: float,
        Papp_ms:  float,
        D_eff:    float,
        dt:       float,
    ):
        self.axis     = axis
        self.position = position
        self.Papp_ms  = Papp_ms
        self.D_eff    = D_eff
        self.dt       = dt

        # crossing probability per attempted step — Δt-invariant
        # derived from Papp × sqrt(π Δt / D_eff)
        # clamped to [0, 1]
        self.permeability = float(np.clip(
            Papp_ms * np.sqrt(np.pi * dt / max(D_eff, 1e-30)),
            0.0, 1.0
        ))
        if self.permeability >= 1.0:
            print(f"⚠ WARNING: Membrane permeability for axis {self.axis} clipped to 1.0. "
                f"Results will be timestep-dependent. Consider reducing dt.")

        self.n_attempts  = 0
        self.n_crossings = 0

    def apply(self, old_positions: np.ndarray, new_positions: np.ndarray) -> np.ndarray:
        """
        Check which particles tried to cross the membrane.
        Allow crossing with probability = self.permeability.
        Reflect blocked particles back symmetrically.

        Parameters
        ----------
        old_positions : np.ndarray (n_particles, 3) — positions before step
        new_positions : np.ndarray (n_particles, 3) — positions after step

        Returns
        -------
        new_positions : np.ndarray — crossings allowed or reflected
        """
        ax  = self.axis
        pos = self.position

        old_side = old_positions[:, ax] >= pos
        new_side = new_positions[:, ax] >= pos
        attempted   = old_side != new_side
        n_attempted = int(np.sum(attempted))

        if n_attempted == 0:
            return new_positions

        self.n_attempts += n_attempted

        rolls   = np.random.uniform(0.0, 1.0, size=n_attempted)
        blocked = rolls >= self.permeability

        blocked_idx = np.where(attempted)[0][blocked]
        new_positions[blocked_idx, ax] = 2.0 * pos - new_positions[blocked_idx, ax]

        self.n_crossings += n_attempted - int(np.sum(blocked))
        return new_positions

    @property
    def crossing_rate(self) -> float:
        """Fraction of crossing attempts that succeeded."""
        if self.n_attempts == 0:
            return 0.0
        return self.n_crossings / self.n_attempts

    def __repr__(self) -> str:
        return (
            f"Membrane(axis={self.axis}, pos={self.position:.2e}, "
            f"Papp={self.Papp_ms:.2e} m/s, P_cross={self.permeability:.4f})"
        )