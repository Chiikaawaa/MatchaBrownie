import numpy as np
from dataclasses import dataclass, field
from typing import Optional


def _km_to_si(km_um: float) -> float:
    return km_um * 1e-3


@dataclass
class TransporterParams:
    name:              str
    gene:              str
    direction:         str            # "efflux" or "influx"
    axis:              int            # transport axis (0=x, 1=y, 2=z)
    Km_um:             float          # Michaelis constant (µM)
    Vmax_ms:           float          # max drift velocity (m/s)
    Ki_um:             Optional[float] = None
    inhibitor_conc_um: float          = 0.0
    expression_scale:  float          = 1.0
    free_fraction:     float          = 1.0
    
    particles_per_mole: float = 1.0 / 6.02214076e23   

    _Km_si: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.direction not in ("efflux", "influx"):
            raise ValueError(f"direction must be 'efflux' or 'influx', got '{self.direction}'")
        if not (0.0 <= self.free_fraction <= 1.0):
            raise ValueError(f"free_fraction must be in [0, 1], got {self.free_fraction}")
        if self.expression_scale < 0:
            raise ValueError(f"expression_scale must be >= 0")
        if self.Vmax_ms <= 0:
            raise ValueError(f"Vmax_ms must be > 0")
        if self.particles_per_mole <= 0:
            raise ValueError(f"particles_per_mole must be > 0")
        self._Km_si = _km_to_si(self.Km_um)

    @property
    def Km_app(self) -> float:
        
        if self.Ki_um is None or self.inhibitor_conc_um == 0.0:
            return self._Km_si
        return self._Km_si * (1.0 + _km_to_si(self.inhibitor_conc_um) / _km_to_si(self.Ki_um))

    def __repr__(self):
        return (f"TransporterParams({self.name}, {self.direction}, "
                f"Km={self.Km_um} µM, Vmax={self.Vmax_ms:.2e} m/s, "
                f"expr={self.expression_scale:.2f}, "
                f"particles_per_mole={self.particles_per_mole:.3e})")


class ActiveTransporter:
    
    def __init__(
        self,
        params:            TransporterParams,
        membrane_position: float,
        membrane_side:     str   = "left",
        activity_zone:     Optional[float] = None,  # metres; None → sqrt(2 D dt)
        D_eff:             float = 1e-12,            # used when activity_zone is None
        dt:                float = 1e-4,             # used when activity_zone is None
    ):
        if membrane_side not in ("left", "right"):
            raise ValueError("membrane_side must be 'left' or 'right'")
        self.params            = params
        self.membrane_position = membrane_position
        self.membrane_side     = membrane_side
        self.D_eff             = D_eff
        self.dt                = dt

       
        if activity_zone is None:
            self._activity_zone_m = np.sqrt(2.0 * max(D_eff, 1e-30) * dt)
        else:
            self._activity_zone_m = float(activity_zone)

        self.n_steps_applied       = 0
        self.total_drift_magnitude = 0.0
        self._drift_history        = []

    def check_concentration_scale(
        self,
        positions:          np.ndarray,
        concentration_field,
        Km_um:              float,
        label:              str = "",
    ) -> None:
        
        if concentration_field is None:
            print(f"[transporter{' '+label if label else ''}] No concentration field — skipping scale check.")
            return
        C_p = self._local_concentration(positions, concentration_field)
        C_m = C_p * self.params.particles_per_mole           # mol/m³
        C_um_max = C_m.max() * 1e3                           # mol/m³ → µM
        Km_app_um = self.params.Km_app * 1e3                 # mol/m³ → µM
        saturation_max = C_m.max() / (self.params.Km_app + C_m.max() + 1e-30)
        tag = f"[{self.params.name}{' '+label if label else ''}]"
        print(
            f"{tag} C_local_max = {C_um_max:.3e} µM  |  "
            f"Km_app = {Km_app_um:.3e} µM  |  "
            f"max saturation = {saturation_max:.3f}  |  "
            f"particles_per_mole = {self.params.particles_per_mole:.3e}"
        )
        if C_um_max < Km_app_um * 0.01:
            print(
                f"   C_local ≪ Km: transporter is deep in linear regime. "
                f"Increase particles_per_mole by ~{Km_app_um / max(C_um_max, 1e-30):.0e}× "
                f"to reach Km."
            )


    def _local_concentration(self, positions: np.ndarray, concentration_field=None) -> np.ndarray:
        """Return concentration (particle counts / m³) at each particle's voxel."""
        if concentration_field is None:    
            raise ValueError(
                    "concentration field is required for transporter kinetics"
                    )
        cf      = concentration_field
        half    = cf.box_size / 2
        indices = np.clip(
            np.floor((positions + half) / cf.voxel_size).astype(int),
            0, cf.n_voxels - 1
        )
        return cf.grid[indices[:, 0], indices[:, 1], indices[:, 2]] / (cf.voxel_size ** 3)

    def _activity_mask(self, positions: np.ndarray, box_size: float) -> np.ndarray:
        
        pos        = positions[:, self.params.axis]
        mp         = self.membrane_position
        zone_width = self._activity_zone_m

        if self.membrane_side == "left":
            return (pos >= mp - zone_width) & (pos < mp)
        return (pos > mp) & (pos <= mp + zone_width)

    def compute_drift(
        self,
        positions:          np.ndarray,
        dt:                 float,
        box_size:           float,
        concentration_field = None,
    ) -> np.ndarray:
        
        drift = np.zeros((len(positions), 3))
        mask  = self._activity_mask(positions, box_size)
        if not np.any(mask):
            return drift

        C_local_particles = self._local_concentration(positions[mask], concentration_field)

        
        C_local_molar = C_local_particles * self.params.particles_per_mole

        Km_app     = self.params.Km_app   
        saturation = C_local_molar / (Km_app + C_local_molar)

       
        v_ms = (
            self.params.Vmax_ms
            * saturation
            * self.params.free_fraction
            * self.params.expression_scale
        )
        sign = 0.0
        if self.params.direction == "efflux":
            sign = -1.0 if self.membrane_side == "left" else 1.0
        else:
            sign = 1.0 if self.membrane_side == "left" else -1.0
        
        drift[mask, self.params.axis] = sign * v_ms * dt

        mag = np.abs(drift[mask, self.params.axis])
        self.total_drift_magnitude += float(mag.sum())
        self._drift_history.append(float(mag.mean()) if len(mag) > 0 else 0.0)
        self.n_steps_applied += 1

        return drift

    def get_stats(self) -> dict:
        if self.n_steps_applied == 0:
            return {"n_steps_applied": 0, "activity_zone_m": self._activity_zone_m}
        hist = np.array(self._drift_history)
        return {
            "n_steps_applied":     self.n_steps_applied,
            "mean_drift_per_step": float(hist.mean()),
            "max_drift_per_step":  float(hist.max()),
            "total_drift":         self.total_drift_magnitude,
            "activity_zone_m":     self._activity_zone_m,
        }

    def reset(self):
        self.n_steps_applied       = 0
        self.total_drift_magnitude = 0.0
        self._drift_history        = []

    def __repr__(self):
        return (f"ActiveTransporter({self.params.name}, {self.params.direction}, "
                f"axis={self.params.axis}, side={self.membrane_side}, "
                f"zone_width={self._activity_zone_m:.2e} m, "
                f"particles_per_mole={self.params.particles_per_mole:.3e})")