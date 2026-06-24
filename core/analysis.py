import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationSummary:
   

    drug_name:   str
    tissue_name: str
    drugbank_id: str
    D_eff:       float

    msd_mean: np.ndarray
    msd_std:  np.ndarray
    msd_time: np.ndarray

    crossing_rate: float
    n_crossed:     int
    n_total:       int

    fpt_mean:         Optional[float]
    fpt_median:       Optional[float]
    fpt_std:          Optional[float]
    fpt_ci_lower:     Optional[float]
    fpt_ci_upper:     Optional[float]
    fpt_t10:          Optional[float]
    fpt_t90:          Optional[float]
    fpt_t95:          Optional[float]
    fpt_distribution: Optional[str]
    n_arrived:        int
    warnings: list[str]


_FIT_PVALUE_THRESHOLD = 0.05
_BOOTSTRAP_N          = 2000     

def _bootstrap_ci(
    times:  np.ndarray,
    n_boot: int   = _BOOTSTRAP_N,
    ci:     float = 0.95,
    seed:   int   = 42,
) -> tuple[float, float]:
    
    rng        = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(times, size=len(times), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1.0 - alpha))),
    )


def fit_distribution(times: np.ndarray) -> str:
    
    distributions = {
        "inverse_gaussian": stats.invgauss, #ideal for fpt with v>0 
        "lognormal":        stats.lognorm,  #realistic for high P_app
        "weibull":          stats.weibull_min,  #flexible, seen when P_app is low
        "gamma":            stats.gamma,   #generalization of exponential, can capture various shapes
        "exponential":      stats.expon, #not ideal for FPTs but included for completeness
        "normal":           stats.norm, #not ideal for FPTs but included for completeness
        "Levy":             stats.levy, #ideal for fpt with v=0
        "pareto":           stats.pareto, #general heavy tailed distribution
    }

    best_name   = None
    best_stat   = np.inf
    best_pvalue = 0.0

    for name, dist in distributions.items():
        try:
            params       = dist.fit(times, floc=0)
            stat, pvalue = stats.kstest(times, dist.cdf, args=params)
            if stat < best_stat:
                best_stat   = stat
                best_name   = name
                best_pvalue = pvalue
        except Exception:
            continue

    if best_name is None:
        return "unknown"
    if best_pvalue < _FIT_PVALUE_THRESHOLD:
        return "poor fit"
    return best_name


def summarise(
    drug,
    tissue,
    D_eff:          float,
    msd_result:     dict,
    membrane,
    fpt_tracker,
    extra_warnings: list = None,
) -> "SimulationSummary":
    all_warnings = list(drug.warnings)
    if extra_warnings:
        all_warnings.extend(extra_warnings)

    fpt_stats = fpt_tracker.get_stats() if fpt_tracker else {}

    times         = fpt_tracker.get_times() if fpt_tracker else np.array([])
    times_nonzero = times[times > 0]

    dist_name = fit_distribution(times_nonzero) if len(times_nonzero) >= 10 else None

    
    if len(times_nonzero) >= 10:
        ci_lower, ci_upper = _bootstrap_ci(times_nonzero)
    else:
        ci_lower = fpt_stats.get("ci_lower")
        ci_upper = fpt_stats.get("ci_upper")

    n_arrived = fpt_stats.get("n_arrived", 0)
    n_total   = (
        fpt_tracker.n_particles_total
        if fpt_tracker is not None and hasattr(fpt_tracker, "n_particles_total")
        else n_arrived
    )

    return SimulationSummary(
        drug_name         = drug.name,
        tissue_name       = tissue.name,
        drugbank_id       = drug.drugbank_id,
        D_eff             = D_eff,
        msd_mean          = msd_result.get("mean"),
        msd_std           = msd_result.get("std"),
        msd_time          = msd_result.get("time"),
        crossing_rate     = membrane.crossing_rate if membrane else 0.0,
        n_crossed         = membrane.n_crossings   if membrane else 0,
        n_total           = n_total,
        fpt_mean          = fpt_stats.get("mean"),
        fpt_median        = fpt_stats.get("median"),
        fpt_std           = fpt_stats.get("std"),
        fpt_ci_lower      = ci_lower,
        fpt_ci_upper      = ci_upper,
        fpt_t10           = fpt_stats.get("t_10pct"),
        fpt_t90           = fpt_stats.get("t_90pct"),
        fpt_t95           = fpt_stats.get("t_95pct"),
        fpt_distribution  = dist_name,
        n_arrived         = n_arrived,
        warnings          = all_warnings,
    )


def print_summary(s: SimulationSummary):
    width = 53
    print("=" * width)
    print("MatchaBrownie — Simulation Summary")
    print("=" * width)
    print(f"  Drug:          {s.drug_name}  ({s.drugbank_id})")
    print(f"  Tissue:        {s.tissue_name}")
    print(f"  D_eff:         {s.D_eff:.3e} m²/s")
    print("-" * width)
    print(f"  Particles total:          {s.n_total}")
    print(f"  Particles arrived:        {s.n_arrived}")
    print(f"  Membrane crossing rate:   {s.crossing_rate:.4f}")
    print("-" * width)
    if s.fpt_mean is not None:
        print(f"  Mean onset time:          {s.fpt_mean:.3f} s")
        print(f"  Median onset time:        {s.fpt_median:.3f} s")
        print(f"  Std deviation:            {s.fpt_std:.3f} s")
        print(f"  95% CI (bootstrap):       [{s.fpt_ci_lower:.3f}, {s.fpt_ci_upper:.3f}] s")
        print(f"  10th percentile:          {s.fpt_t10:.3f} s")
        print(f"  90th percentile:          {s.fpt_t90:.3f} s")
        print(f"  95th percentile:          {s.fpt_t95:.3f} s")
        print(f"  Distribution fit:         {s.fpt_distribution}")
    print("-" * width)
    if s.warnings:
        for warning in s.warnings:
            print(f"  ⚠ {warning}")
    else:
        print("  No warnings.")
    print("=" * width)