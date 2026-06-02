import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WallConfig:
    """
    Per-wall boundary condition.

    Parameters
    ----------
    mode : str
        "reflect"  — hard elastic reflection (default)
        "absorb"   — particle is removed from the simulation on contact.
                     Models systemic clearance by blood flow or lymphatics
                     (sink condition). The particle's position is clamped to
                     the wall plane so it cannot appear outside the box in
                     downstream arrays.
        "membrane" — probabilistic crossing; blocked particles are reflected.
    permeability : float
        Crossing probability in [0, 1]. Only used when mode="membrane".
        0.0 = perfectly reflecting, 1.0 = perfectly absorbing.
        Derived from Papp the same way Membrane does it:
            permeability = Papp_ms * sqrt(π * dt / D_eff)
        Pass the pre-computed value here, or use WallConfig.from_papp().

    Sink-condition recipe (open physiological compartment)
    ------------------------------------------------------
    In living tissue the far boundary represents systemic clearance, not a
    hard wall. Use an absorbing exit so particles are cleared rather than
    reflected back, which would create artificial "echoes" and bias the FPT
    distribution toward Weibull rather than inverse-Gaussian:

        walls = {
            "z_lo": WallConfig(mode="reflect"),   # donor / source side
            "z_hi": WallConfig(mode="absorb"),    # receiver / clearance side
        }
        geometry = BoxGeometry(bounds=bounds, walls=walls)

    Vacuum-test tip
    ---------------
    To verify the Brownian engine is mathematically sound, run with
    geometry=None and membrane=None.  MSD should track 6Dt exactly.
    Any deviation in that mode points to a bug in the stepper, not the BCs.
    """

    mode:         str   = "reflect"
    permeability: float = 0.0

    def __post_init__(self):
        if self.mode not in ("reflect", "absorb", "membrane"):
            raise ValueError(
                f"mode must be 'reflect', 'absorb', or 'membrane', got '{self.mode}'"
            )
        if not (0.0 <= self.permeability <= 1.0):
            raise ValueError(
                f"permeability must be in [0, 1], got {self.permeability}"
            )

    @staticmethod
    def from_papp(Papp_ms: float, dt: float, D_eff: float) -> "WallConfig":
        """
        Build a membrane WallConfig from physical apparent permeability.

        Uses the same Δt-invariant formula as core.membrane.Membrane:
            P_cross = Papp_ms × sqrt(π × dt / D_eff)

        Parameters
        ----------
        Papp_ms : apparent permeability in m/s
        dt      : simulation timestep in seconds
        D_eff   : effective diffusion coefficient in m²/s
        """
        p = float(np.clip(
            Papp_ms * np.sqrt(np.pi * dt / max(D_eff, 1e-30)),
            0.0, 1.0,
        ))
        return WallConfig(mode="membrane", permeability=p)


# Six faces in order: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
_FACE_LABELS = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")

# Default: all walls reflect
_DEFAULT_WALLS: dict[str, WallConfig] = {
    label: WallConfig(mode="reflect") for label in _FACE_LABELS
}


class BoxGeometry:
    """
    3D rectangular boundary with per-wall boundary conditions.

    Each of the six faces can independently be:
        "reflect"  — hard elastic reflection
        "absorb"   — particle removed on contact (clamped to wall plane)
        "membrane" — probabilistic crossing (blocked → reflected)

    Parameters
    ----------
    bounds : array-like, shape (3, 2)
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in metres.
    walls : dict[str, WallConfig], optional
        Override any subset of the six faces.
        Keys: "x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi".
        Unspecified faces default to WallConfig(mode="reflect").

    Examples
    --------
    # All-reflecting (original behaviour)
    geom = BoxGeometry(bounds)

    # Absorbing exit at +z (sink condition), reflecting everywhere else
    geom = BoxGeometry(bounds, walls={"z_hi": WallConfig(mode="absorb")})

    # Probabilistic membrane at +z, built from Papp
    geom = BoxGeometry(bounds, walls={
        "z_hi": WallConfig.from_papp(Papp_ms=1e-6, dt=dt, D_eff=D_eff)
    })
    """

    def __init__(
        self,
        bounds,
        walls: Optional[dict] = None,
    ):
        self.bounds = np.array(bounds, dtype=float)   # shape (3, 2)

        # Merge caller-supplied overrides with defaults
        self.walls: dict[str, WallConfig] = dict(_DEFAULT_WALLS)
        if walls:
            for key, cfg in walls.items():
                if key not in _FACE_LABELS:
                    raise ValueError(
                        f"Unknown wall key '{key}'. Valid keys: {_FACE_LABELS}"
                    )
                self.walls[key] = cfg

        # Counters — one entry per face
        self.n_attempts:  dict[str, int] = {k: 0 for k in _FACE_LABELS}
        self.n_crossings: dict[str, int] = {k: 0 for k in _FACE_LABELS}
        self.n_absorbed:  dict[str, int] = {k: 0 for k in _FACE_LABELS}

        # Tracks which particles are still active (not absorbed)
        self._active: Optional[np.ndarray] = None   # set lazily on first apply()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        old_positions: np.ndarray,
        new_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Enforce boundary conditions for all six walls.

        Parameters
        ----------
        old_positions : (n_particles, 3) — positions before the step
        new_positions : (n_particles, 3) — proposed positions after the step

        Returns
        -------
        new_positions : (n_particles, 3) — positions after BCs applied.
            Absorbed particles are clamped to the wall plane and marked
            inactive in self._active; use self.active_mask to filter them.
        """
        n = len(new_positions)
        if self._active is None:
            self._active = np.ones(n, dtype=bool)

        positions = new_positions.copy()

        # Dimension → (lo_face_key, hi_face_key)
        face_pairs = [
            ("x_lo", "x_hi"),
            ("y_lo", "y_hi"),
            ("z_lo", "z_hi"),
        ]

        for dim, (lo_key, hi_key) in enumerate(face_pairs):
            lo, hi = self.bounds[dim]

            self._apply_wall(
                positions, old_positions, dim,
                wall_pos=lo, over_wall=(positions[:, dim] < lo),
                face_key=lo_key,
            )
            self._apply_wall(
                positions, old_positions, dim,
                wall_pos=hi, over_wall=(positions[:, dim] > hi),
                face_key=hi_key,
            )

        return positions

    def is_inside(self, positions: np.ndarray) -> np.ndarray:
        """Boolean mask — True if particle is inside the box."""
        inside = np.ones(len(positions), dtype=bool)
        for dim in range(3):
            lo, hi = self.bounds[dim]
            inside &= (positions[:, dim] >= lo) & (positions[:, dim] <= hi)
        return inside

    @property
    def active_mask(self) -> np.ndarray:
        """Boolean mask of particles that have not been absorbed."""
        if self._active is None:
            raise RuntimeError("apply() has not been called yet.")
        return self._active.copy()

    @property
    def volume(self) -> float:
        """Volume of the box in m³."""
        dims = self.bounds[:, 1] - self.bounds[:, 0]
        return float(np.prod(dims))

    def crossing_rate(self, face: str) -> float:
        """Fraction of crossing attempts that succeeded for a given face."""
        if face not in _FACE_LABELS:
            raise ValueError(f"Unknown face '{face}'. Valid: {_FACE_LABELS}")
        attempts = self.n_attempts[face]
        return self.n_crossings[face] / attempts if attempts > 0 else 0.0

    def reset_counters(self):
        """Reset attempt / crossing / absorbed counters and active mask."""
        self.n_attempts  = {k: 0 for k in _FACE_LABELS}
        self.n_crossings = {k: 0 for k in _FACE_LABELS}
        self.n_absorbed  = {k: 0 for k in _FACE_LABELS}
        self._active     = None

    def summary(self) -> dict:
        """Return a dict of per-face stats for inspection / logging."""
        return {
            face: {
                "mode":          self.walls[face].mode,
                "permeability":  self.walls[face].permeability,
                "n_attempts":    self.n_attempts[face],
                "n_crossings":   self.n_crossings[face],
                "n_absorbed":    self.n_absorbed[face],
                "crossing_rate": self.crossing_rate(face),
            }
            for face in _FACE_LABELS
        }

    def __repr__(self) -> str:
        b = self.bounds
        modes = {k: self.walls[k].mode for k in _FACE_LABELS}
        return (
            f"BoxGeometry("
            f"x=[{b[0,0]:.2e},{b[0,1]:.2e}], "
            f"y=[{b[1,0]:.2e},{b[1,1]:.2e}], "
            f"z=[{b[2,0]:.2e},{b[2,1]:.2e}], "
            f"walls={modes})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_wall(
        self,
        positions:     np.ndarray,
        old_positions: np.ndarray,
        dim:           int,
        wall_pos:      float,
        over_wall:     np.ndarray,
        face_key:      str,
    ):
        """Apply the BC for one wall face to the particles that crossed it."""
        # Only operate on active particles that actually crossed this wall
        hit_idx = np.where(over_wall & self._active)[0]
        if len(hit_idx) == 0:
            return

        cfg = self.walls[face_key]

        if cfg.mode == "reflect":
            self._reflect(positions, hit_idx, dim, wall_pos)

        elif cfg.mode == "absorb":
            # Clamp position to the wall plane so the particle stays inside the
            # grid and doesn't corrupt concentration-field binning or MSD.
            # Previously positions were left out-of-bounds, which caused voxel
            # index overflows and inflated MSD after absorption.
            positions[hit_idx, dim] = wall_pos
            self._active[hit_idx]   = False
            self.n_absorbed[face_key] += len(hit_idx)

        elif cfg.mode == "membrane":
            self.n_attempts[face_key] += len(hit_idx)

            rolls   = np.random.uniform(0.0, 1.0, size=len(hit_idx))
            blocked = hit_idx[rolls >= cfg.permeability]
            allowed = hit_idx[rolls <  cfg.permeability]

            # Blocked → reflect back
            self._reflect(positions, blocked, dim, wall_pos)

            # Allowed → crossed; keep new position (outside box by design)
            self.n_crossings[face_key] += len(allowed)

    @staticmethod
    def _reflect(
        positions: np.ndarray,
        idx:       np.ndarray,
        dim:       int,
        wall_pos:  float,
    ):
        """Single-pass elastic reflection about wall_pos."""
        if len(idx) == 0:
            return
        positions[idx, dim] = 2.0 * wall_pos - positions[idx, dim]