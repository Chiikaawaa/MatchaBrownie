import numpy as np


class Membrane:
    
    def __init__(
        self,
        axis:     int,
        position: float,
        Papp_ms:  float,
        D_eff:    float,
        dt:       float,
        box_size: float,
    ):
        self.axis     = axis
        self.position = position
        self.Papp_ms  = Papp_ms
        self.D_eff    = D_eff
        self.dt       = dt

        
        self.permeability = float(np.clip(
            Papp_ms * np.sqrt(np.pi * dt / D_eff),
            0.0, 1.0
        ))
        if self.permeability >= 1.0:
            print(f" WARNING: Membrane permeability for axis {self.axis} clipped to 1.0. "
                f"Results will be timestep-dependent. Consider reducing dt.")

        self.n_attempts  = 0
        self.n_crossings = 0

    def apply(self, old_positions: np.ndarray, new_positions: np.ndarray) -> np.ndarray:
        
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
        if self.n_attempts == 0:
            return 0.0
        return self.n_crossings / self.n_attempts

    def __repr__(self) -> str:
        return (
            f"Membrane(axis={self.axis}, pos={self.position:.2e}, "
            f"Papp={self.Papp_ms:.2e} m/s, P_cross={self.permeability:.4f})"
        )