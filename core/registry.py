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
        self.box_size        = float(data["box_size_um"]) * 1e-6
        self.transporter_expression        = data.get("transporter_expression", {})
        self.transporter_expression_source = data.get("transporter_expression_source", "not set")

    def get_expression_scale(self, gene: str) -> float:
       
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
        self.Papp_cms         = float(data.get("Papp_cms")) if data.get("Papp_cms") is not None else None
        self.Papp_source     = data.get("Papp_source", "default")
        
        if self.Papp_cms is not None:
            self.Papp_ms = self.Papp_cms * 1e-2
        else:
            self.Papp_ms = None
        
        self.warnings = []
        if self.pb_source == "default":
            self.warnings.append(
                f"protein_binding estimated at 0.5 — verify at "
                f"drugbank.com/drugs/{self.name}"
            )
        if self.Papp_ms is None:
            self.warnings.append(
                f"Papp_ms not available in drug Yaml"
            )

    def __repr__(self):
        return f"DrugParams({self.name}, MW={self.MW}, logP={self.logP}, pKa={self.pKa}, protein_binding={self.protein_binding}, Papp_ms={self.Papp_ms})"


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
       
        kT            = 1.380649e-23 * 310.15   
        D_base        = kT / (6 * np.pi * tissue.viscosity * drug.radius)
        free_fraction = 1.0 - drug.protein_binding
        trial_factor:float  = tissue.permeability
        return D_base * (1.00/ tissue.tortuosity ** 2) * free_fraction

    @staticmethod
    def calibrate_particles_per_mole(
        dose_uM:      float,
        box_size_m:   float,
        n_particles:  int,
    ) -> float:

        dose_mol_per_m3  = dose_uM * 1e-3          
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

        from core.transporter import TransporterParams, ActiveTransporter

        if gene not in GENE_META:
            raise ValueError(f"Unknown gene '{gene}'. Known: {list(GENE_META.keys())}")

        name, direction  = GENE_META[gene]
        expression_scale = tissue.get_expression_scale(gene)
        free_fraction    = 1.0 - drug.protein_binding

        if n_particles is not None and dose_uM is not None:
            particles_per_mole = Registry.calibrate_particles_per_mole(
                dose_uM=dose_uM,
                box_size_m=tissue.box_size,
                n_particles=n_particles,
            )
        else:
            particles_per_mole = 1.0 / 6.02214076e23   
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
        )