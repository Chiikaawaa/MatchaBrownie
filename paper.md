# Drug Transport Simulator: Membrane Permeability Controls First Passage Time Distribution

## Abstract
Drug transport across biological membranes is a critical determinant of pharmacokinetic onset, yet stochastic models that resolve tissue specific permeability remain underdeveloped. MatchaBrownie is a particle based Brownian motion engine for simulating drug transport in isolated tissue system and tracking first passage time and related statistical properties. The model integrates drug properties from DrugBank with tissue specific parameters from Human Protein Atlas. 

The simulator uses Euler-Maruyama method and has a C++ accelerated core achieving 3× speed up over pure python (CPU only). Drug transport is governed by permeability coefficients calculated using drug and tissue specific parameters rather than fixed translation probabilities. In addition to passive diffusion the core supports active transport mechanisms in a configurable active mask region modeled using Michaelis-Menten kinetics which enables both efflux and influx transport.

Its physics is validated against Sidney Redner's text "A Guide to First Passage Processes". For drift driven motion simulated first passage times match the theoretical Inverse Gaussian distribution. For drift free motion they match the Levy distribution.

MatchaBrownie provides a computationally efficient platform for studying tissue specific drug transport dynamics by integrating passive diffusion, membrane permeability and active transport within a unified framework. The simulator can be used to prioritize candidate compounds for experimental diffusion studies, potentially reducing the number of costly laboratory experiments required during early stage pharmacokinetic analysis.


## 1. Introduction
(Write this third. Leave blank for now.)

## 2. Methods

### 2.1 Stochastic Simulator
(Describe simulator architecture)

### 2.2 Effective Diffusivity
(Explain D_eff formula)

### 2.3 Membrane Model
(permeability)

## 3. Results

### 3.1 Validation Against Redner Theory
(Inverse Gaussian result)

### 3.2 Case 1: Low Permeability

#### example 1: Transport of Lidocaine across Muscle tissue.
Drug:            lidocaine (DB00281)
Tissue:          muscle
MW:              234.3373 Da
logP:            2.44
pKa:             8.01
Protein binding: 80%
D_eff:           2.307e-11 m²/s
P-gp expression: 0.0021
Membrane P_cross:     0.148543
Step sigma:           151.8728 nm
Box size:             40.0 μm
FPT threshold:        18.0 μm
  Particles arrived at threshold: 263
  Membrane crossings (internal):  8095
  Absorbed at z_hi:  247
  Active particles remaining:     253
Mean step size:    119.8133 nm
Expected sigma:    151.8728 nm
Ratio:             0.7889  (expect ~0.80)
FPT > 0:           263
Min FPT:           2.8550 s
Median FPT:        13.2755 s
Internal membrane crossing rate: 0.149142
=====================================================
MatchaBrownie — Simulation Summary
=====================================================
  Drug:          lidocaine  (DB00281)
  Tissue:        muscle
  D_eff:         2.307e-11 m²/s
-----------------------------------------------------
  Particles total:          500
  Particles arrived:        263
  Membrane crossing rate:   0.1491
-----------------------------------------------------
  Mean onset time:          13.757 s
  Median onset time:        13.276 s
  Std deviation:            5.859 s
  95% CI (bootstrap):       [13.027, 14.473] s
  10th percentile:          6.255 s
  90th percentile:          22.149 s
  95th percentile:          23.128 s
  Distribution fit:         weibull
============================================================
TOTAL RUNTIME: 1263.40 seconds
============================================================

#### example 2:

### 3.3 Case 3: Moderate Permeability

#### example 1:

#### example 2:

### 3.4 Case 3: High Permeability

#### example 1:

#### example 2:

## 4. Discussion
(Leave blank for now)

## 5. Conclusion
(Leave blank for now)

## References
(Add on the way)