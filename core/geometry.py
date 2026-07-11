import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sim_core
from sim_core import step_core
from sim_core import apply_geometry

@dataclass
class WallConfig:
    

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
    def from_papp(Papp_ms: float, dt: float, D_eff: float, box_size: float) -> "WallConfig":
        
        p = float(np.clip(
            Papp_ms * np.sqrt(np.pi * dt / D_eff),
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
    

    def __init__(
        self,
        bounds,
        walls: Optional[dict] = None,
    ):
        self.bounds = np.array(bounds, dtype=float)   

        
        self.walls: dict[str, WallConfig] = dict(_DEFAULT_WALLS)
        if walls:
            for key, cfg in walls.items():
                if key not in _FACE_LABELS:
                    raise ValueError(
                        f"Unknown wall key '{key}'. Valid keys: {_FACE_LABELS}"
                    )
                self.walls[key] = cfg

        
        self.n_attempts:  dict[str, int] = {k: 0 for k in _FACE_LABELS}
        self.n_crossings: dict[str, int] = {k: 0 for k in _FACE_LABELS}
        self.n_absorbed:  dict[str, int] = {k: 0 for k in _FACE_LABELS}

        self._active: Optional[np.ndarray] = None   
    

    def apply(self, old_positions: np.ndarray, new_positions: np.ndarray) -> np.ndarray:
        n=len(new_positions)
        if self._active is None:
            self._active = np.ones(n, dtype=bool)
        positions = new_positions.copy().astype(np.float64)
        bounds = self.bounds.astype(np.float64)
        mode_map = {"reflect": 0, "absorb": 1, "membrane": 2}
        wall_modes = np.array([
            mode_map[self.walls[face].mode] 
            for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
        ], dtype=np.int32)
        permeabilities = np.array([
            self.walls[face].permeability 
            for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
        ], dtype=np.float64)
    
        n_att = np.zeros(6, dtype=np.int32)
        n_cross = np.zeros(6, dtype=np.int32)
        n_abs = np.zeros(6, dtype=np.int32)
        sim_core.apply_geometry(
            positions, self._active, bounds, wall_modes, permeabilities,
            n_att, n_cross, n_abs
        )
    
    # Update Python counters from C++
        face_labels = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
        for i, face in enumerate(face_labels):
            self.n_attempts[face] += int(n_att[i])
            self.n_crossings[face] += int(n_cross[i])
            self.n_absorbed[face] += int(n_abs[i])
    
        return positions

    def is_inside(self, positions: np.ndarray) -> np.ndarray:
        inside = np.ones(len(positions), dtype=bool)
        for dim in range(3):
            lo, hi = self.bounds[dim]
            inside &= (positions[:, dim] >= lo) & (positions[:, dim] <= hi)
        return inside

    @property
    def active_mask(self) -> np.ndarray:
        if self._active is None:
            raise RuntimeError("apply() has not been called yet.")
        return self._active.copy()

    @property
    def volume(self) -> float:
        dims = self.bounds[:, 1] - self.bounds[:, 0]
        return float(np.prod(dims))

    def crossing_rate(self, face: str) -> float:
        if face not in _FACE_LABELS:
            raise ValueError(f"Unknown face '{face}'. Valid: {_FACE_LABELS}")
        attempts = self.n_attempts[face]
        return self.n_crossings[face] / attempts if attempts > 0 else 0.0

    def reset_counters(self):
        self.n_attempts  = {k: 0 for k in _FACE_LABELS}
        self.n_crossings = {k: 0 for k in _FACE_LABELS}
        self.n_absorbed  = {k: 0 for k in _FACE_LABELS}
        self._active     = None

    def summary(self) -> dict:
        
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


    def _apply_wall(
        self,
        positions:     np.ndarray,
        old_positions: np.ndarray,
        dim:           int,
        wall_pos:      float,
        over_wall:     np.ndarray,
        face_key:      str,
    ):
        hit_idx = np.where(over_wall & self._active)[0]
        if len(hit_idx) == 0:
            return

        cfg = self.walls[face_key]

        if cfg.mode == "reflect":
            self._reflect(positions, hit_idx, dim, wall_pos)

        elif cfg.mode == "absorb":
            
            positions[hit_idx, dim] = wall_pos
            self._active[hit_idx]   = False
            self.n_absorbed[face_key] += len(hit_idx)

        elif cfg.mode == "membrane":
            self.n_attempts[face_key] += len(hit_idx)

            rolls   = np.random.uniform(0.0, 1.0, size=len(hit_idx))
            blocked = hit_idx[rolls >= cfg.permeability]
            allowed = hit_idx[rolls <  cfg.permeability]

            
            self._reflect(positions, blocked, dim, wall_pos)

            self.n_crossings[face_key] += len(allowed)

    @staticmethod
    def _reflect(
        positions: np.ndarray,
        idx:       np.ndarray,
        dim:       int,
        wall_pos:  float,
    ):
        if len(idx) == 0:
            return
        positions[idx, dim] = 2.0 * wall_pos - positions[idx, dim]