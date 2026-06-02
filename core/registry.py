import yaml
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

ALIAS_MAP = {
    "paracetamol": "acetaminophen",
    "tylenol":     "acetaminophen",
    "advil":       "ibuprofen",
    "motrin":      "ibuprofen",
    "valium":      "diazepam",
    "aspirin":     "acetylsalicylic_acid",
    "adrenaline":  "epinephrine",
    "vitamin_c":   "ascorbic_acid",
}

GENE_META = {
    "SLC19A1": ("RFC1",    "influx"),
    "SLC29A1": ("ENT1",    "influx"),
    "SLC46A1": ("PCFT",    "influx"),
    "ABCB1":   ("P-gp",    "efflux"),
    "ABCG2":   ("BCRP",    "efflux"),
    "SLCO1B1": ("OATP1B1", "influx"),
    "SLC22A1": ("OCT1",    "influx"),
    "ABCC2":   ("MRP2",    "efflux"),
}


class TissueParams:
    def __init__(self, data: dict):
        self.name            = data["name"]
        self.description     = data.get("description", "")
        self.D               = float(data["diffusion_coefficient"])
        self.viscosity       = float(data["viscosity"])
        self.permeability    = float(data["permeability"])
        self.tortuosity      = float(data["tortuosity"])
        self.ph              = float(data["ph"])
        self.box_size        = float(data["box_size_um"]) * 1e-6
        self.transporter_expression        = data.get("transporter_expression", {})
        self.transporter_expression_source = data.get("transporter_expression_source", "not set")

    def get_expression_scale(self, gene: str) -> float:
        """Normalised HPA expression scale for a transporter gene (0–1). 0 if absent."""
        return float(self.transporter_expression.get(gene, 0.0))

    def __repr__(self):
        return f"TissueParams({self.name}, D={self.D:.2e}, P={self.permeability})"


class DrugParams:
    def __init__(self, data: dict):
        self.name            = data["name"]
        self.description     = data.get("description", "")
        self.drugbank_id     = data.get("drugbank_id", "")
        self.MW              = float(data["molecular_weight"])
        self.logP            = float(data["logP"]) if data.get("logP") else 1.0
        self.pKa             = float(data["pKa"]) if data.get("pKa") else 7.0
        self.protein_binding = float(data["protein_binding"]) if data.get("protein_binding") else 0.5
        self.pb_source       = data.get("protein_binding_source", "default")
        self.half_life       = data.get("half_life", "unknown")
        self.absorption      = data.get("absorption", "unknown")
        self.metabolism      = data.get("metabolism", "unknown")
        self.indication      = data.get("indication", "unknown")
        self.charge          = int(data.get("charge", 0))
        self.radius          = 0.0483e-9 * (self.MW ** (1/3))

        self.warnings = []
        if self.pb_source == "default":
            self.warnings.append(
                f"protein_binding estimated at 0.5 — verify at "
                f"drugbank.com/drugs/{self.name}"
            )

    def __repr__(self):
        return f"DrugParams({self.name}, MW={self.MW}, logP={self.logP})"


class Registry:

    @staticmethod
    def load_tissue(name: str) -> TissueParams:
        path = DATA_DIR / "tissues" / f"{name}.yaml"
        if not path.exists():
            available = [f.stem for f in (DATA_DIR / "tissues").glob("*.yaml")]
            raise FileNotFoundError(
                f"Tissue '{name}' not found. Available: {available}"
            )
        with open(path, encoding="utf-8") as f:
            return TissueParams(yaml.safe_load(f))

    @staticmethod
    def load_drug(name: str) -> DrugParams:
        resolved = ALIAS_MAP.get(name.lower(), name.lower())
        path     = DATA_DIR / "drugs" / f"{resolved}.yaml"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                drug = DrugParams(yaml.safe_load(f))
            for w in drug.warnings:
                print(f"⚠ {name}: {w}")
            return drug
        available = [f.stem for f in (DATA_DIR / "drugs").glob("*.yaml")]
        raise FileNotFoundError(
            f"Drug '{name}' not found in local database.\n"
            f"Try the DrugBank name directly — search at drugbank.com.\n"
            f"({len(available)} drugs available locally)"
        )

    @staticmethod
    def list_tissues() -> list:
        return sorted([f.stem for f in (DATA_DIR / "tissues").glob("*.yaml")])

    @staticmethod
    def list_drugs() -> list:
        return sorted([f.stem for f in (DATA_DIR / "drugs").glob("*.yaml")])

    @staticmethod
    def compute_D_eff(drug: DrugParams, tissue: TissueParams) -> float:
        """
        Stokes-Einstein corrected for tortuosity squared (Archie's law)
        and free fraction.

            D_eff = kT / (6π η r) × (1 / tortuosity²) × (1 − protein_binding)

        The free-fraction factor is applied here and nowhere else. Transporters
        read free_fraction from DrugParams and apply it to Vmax; they must NOT
        re-apply it to D_eff-derived quantities (e.g. P_cross) or to the drift
        velocity a second time.
        """
        kT            = 1.380649e-23 * 310.15   # body temp 37°C
        D_base        = kT / (6 * np.pi * tissue.viscosity * drug.radius)
        free_fraction = 1.0 - drug.protein_binding
        return D_base * (1.0 / tissue.tortuosity ** 2) * free_fraction

    @staticmethod
    def calibrate_particles_per_mole(
        dose_uM:      float,
        box_size_m:   float,
        n_particles:  int,
    ) -> float:
        """
        Compute the particles_per_mole scaling factor so that the local molar
        concentration seen by Michaelis-Menten kinetics matches a physical dose.

        Without this, 1 particle = 1 molecule, giving absurdly low molar
        concentrations in µm-scale boxes (e.g. ~1e-12 M per particle) and
        preventing transporter saturation at any realistic dose.

        Parameters
        ----------
        dose_uM     : physical dosing concentration in µM
        box_size_m  : edge length of the cubic simulation box in metres
        n_particles : number of simulation particles

        Returns
        -------
        particles_per_mole : float
            Pass this directly to TransporterParams(particles_per_mole=…).

        Example
        -------
        >>> ppm = Registry.calibrate_particles_per_mole(
        ...     dose_uM=10.0, box_size_m=10e-6, n_particles=1000
        ... )
        # 10 µM in a 10 µm box → ~1e-17 mol total
        # 1000 particles → ppm ≈ 1e-20 mol/particle
        """
        dose_mol_per_m3  = dose_uM * 1e-3          # µM → mol/m³
        box_volume_m3    = box_size_m ** 3
        moles_in_box     = dose_mol_per_m3 * box_volume_m3
        return moles_in_box / n_particles

    @staticmethod
    def make_transporter_for_tissue(
        gene:              str,
        tissue:            TissueParams,
        drug:              DrugParams,
        D_eff:             float,
        dt:                float,
        axis:              int   = 2,
        membrane_position: float = 0.0,
        Km_um:             float = 10.0,
        Vmax_ms:           float = 3e-9,
        Ki_um:             float = None,
        inhibitor_conc_um: float = 0.0,
        membrane_side:     str   = "left",
        n_particles:       int   = None,
        dose_uM:           float = None,
    ):
        """
        Build an ActiveTransporter with expression_scale pulled automatically
        from the tissue's HPA RNA data.

        free_fraction is read from drug.protein_binding. It is passed to
        TransporterParams and applied to Vmax inside ActiveTransporter.compute_drift()
        only — it is NOT applied a second time via D_eff or P_cross.

        particles_per_mole is calibrated automatically when both n_particles and
        dose_uM are supplied; otherwise falls back to the legacy 1-molecule default
        and emits a warning.

        Parameters
        ----------
        gene              : HGNC gene symbol — one of ABCB1, ABCG2, SLCO1B1, SLC22A1, ABCC2
        tissue            : TissueParams loaded from registry
        drug              : DrugParams loaded from registry
        D_eff             : effective diffusion coefficient (m²/s) — used to set
                            the activity zone width (sqrt(2 D_eff dt))
        dt                : simulation timestep (s) — used for activity zone
        Km_um             : Michaelis constant in µM (literature value)
        Vmax_ms           : max drift velocity in m/s (typical: 1–10 nm/s)
        Ki_um             : inhibition constant in µM (competitive inhibition)
        inhibitor_conc_um : inhibitor concentration in µM
        membrane_side     : which side of membrane the transporter acts on
        n_particles       : number of simulation particles (needed for calibration)
        dose_uM           : physical dosing concentration in µM (needed for calibration)
        """
        from core.transporter import TransporterParams, ActiveTransporter

        if gene not in GENE_META:
            raise ValueError(f"Unknown gene '{gene}'. Known: {list(GENE_META.keys())}")

        name, direction  = GENE_META[gene]
        expression_scale = tissue.get_expression_scale(gene)
        free_fraction    = 1.0 - drug.protein_binding

        # Calibrate particles_per_mole if the caller supplied dose and count.
        if n_particles is not None and dose_uM is not None:
            particles_per_mole = Registry.calibrate_particles_per_mole(
                dose_uM=dose_uM,
                box_size_m=tissue.box_size,
                n_particles=n_particles,
            )
        else:
            particles_per_mole = 1.0 / 6.02214076e23   # legacy: 1 particle = 1 molecule
            print(
                f"⚠ make_transporter_for_tissue: particles_per_mole not calibrated "
                f"(n_particles or dose_uM not supplied). Transporter will operate in "
                f"the deep-linear MM regime. Pass n_particles and dose_uM for accurate kinetics."
            )

        params = TransporterParams(
            name               = name,
            gene               = gene,
            direction          = direction,
            axis               = axis,
            Km_um              = Km_um,
            Vmax_ms            = Vmax_ms,
            Ki_um              = Ki_um,
            inhibitor_conc_um  = inhibitor_conc_um,
            expression_scale   = expression_scale,
            free_fraction      = free_fraction,
            particles_per_mole = particles_per_mole,
        )

        return ActiveTransporter(
            params            = params,
            membrane_position = membrane_position,
            membrane_side     = membrane_side,
            D_eff             = D_eff,
            dt                = dt,
            # activity_zone left as None → defaults to sqrt(2 D_eff dt)
        )