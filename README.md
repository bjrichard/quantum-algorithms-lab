# Quantum Algorithms Lab

`quantum-algorithms-lab` is a notebook-based collection of reproducible experiments in quantum information, circuit behavior, and variational quantum algorithms using Python and Qiskit.

The repository progresses from foundational state and measurement models to hardware-aware circuit analysis and Variational Quantum Eigensolver (VQE) experiments. Its strongest focus is the effect of finite-shot sampling and optimizer choice on small VQE workflows.

## Research focus

The later experiments examine the question:

> How do ansatz design, hardware constraints, finite-shot sampling, and optimizer choice affect small VQE experiments?

The repository is exploratory rather than a production software library. Notebooks emphasize transparent calculations, compact examples, and interpretation of observed behavior.

## What the repository covers

### Quantum information foundations

- statevectors and density matrices
- projective measurements and positive operator-valued measures (POVMs)
- quantum channels and simple noise models
- circuit and unitary equivalence

### Hardware-aware circuits

- connectivity constraints
- transpilation
- circuit depth and gate-count changes
- comparison of logical and hardware-constrained circuit representations

### Variational quantum algorithms

- end-to-end VQE construction
- exact expectation-value baselines
- finite-shot energy estimation
- repeated-run variability
- comparison of Constrained Optimization by Linear Approximation (COBYLA) and Simultaneous Perturbation Stochastic Approximation (SPSA)
- reusable utilities for compact VQE experiments

## Notebook guide

| Notebook | Focus |
|---|---|
| `00_sanity_check.ipynb` | Environment and Qiskit setup |
| `01_basic_circuits.ipynb` | Basic circuit construction and execution |
| `02_density_matrices.ipynb` | Pure and mixed states using density matrices |
| `03_measurements_and_povms.ipynb` | Projective measurements and POVMs |
| `04_quantum_channels_and_noise.ipynb` | Quantum channels and simple noise effects |
| `06_circuits_and_unitaries.ipynb` | Circuit-unitary relationships and equivalence |
| `07_hardware_constraints_depth_connectivity.ipynb` | Connectivity, transpilation, and circuit depth |
| `08_first_vqe_end_to_end.ipynb` | First complete VQE workflow |
| `09_vqe_noise_and_optimization_sensitivity.ipynb` | Finite-shot variability and optimizer sensitivity |
| `10_vqe_experiments_with_utils.ipynb` | Reusable VQE utilities and repeated-run summaries |

## VQE experiments

The VQE notebooks use a small two-qubit Hamiltonian and a hardware-efficient parameterized ansatz to compare:

- exact expectation values from `StatevectorEstimator`
- finite-shot estimates from Qiskit Aer
- repeated runs under identical nominal settings
- COBYLA and SPSA optimization behavior

A representative saved run in `10_vqe_experiments_with_utils.ipynb` produced:

```text
Exact energy:       -1.4030
COBYLA mean ± std:  -1.3719 ± 0.0598
SPSA mean ± std:    -1.3789 ± 0.0398
```

These values document one small experiment, not a general benchmark. Three repeated runs are sufficient to illustrate sampling and optimization variability but not to establish broad optimizer superiority.

## Reusable VQE utilities

[`src/vqe_utils.py`](src/vqe_utils.py) provides:

- construction of a small hardware-efficient ansatz
- exact VQE evaluation
- finite-shot VQE evaluation
- repeated-run execution
- summary statistics for observed energies

The utilities support the notebooks but are not presented as a stable public application programming interface (API).

## Quick start

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate quantum-algorithms-lab
```

Start JupyterLab:

```bash
jupyter lab
```

Run the notebooks in numerical order. The later VQE notebooks depend on concepts introduced earlier, but each notebook is intended to remain reasonably self-contained.

## Repository structure

```text
notebooks/         # exploratory and reproducible experiments
src/               # reusable VQE utilities
notes/             # supporting notes
environment.yml    # Conda environment definition
```

## Reproducibility

The notebooks retain representative outputs so that results can be inspected without rerunning every experiment.

Exact numerical results from finite-shot and optimizer-based experiments can vary between runs because of:

- measurement sampling
- optimizer initialization
- stochastic optimization steps
- software-version differences

The repository therefore treats individual numerical outputs as examples and emphasizes qualitative behavior and repeated-run summaries.

## Current limitations

The repository is intentionally limited to small simulations and educationally transparent experiments. It does not currently include:

- execution on quantum hardware
- large problem instances
- systematic hyperparameter sweeps
- statistically powered optimizer comparisons
- automated tests for the notebook workflows
- continuous integration
- production-scale package architecture

The reusable utility module would benefit from automated unit tests as the repository develops further.

## Project status

The repository is a completed exploratory laboratory covering quantum-information foundations, hardware-aware circuit behavior, and introductory VQE experimentation.

Future work could include:

- larger repeated-run studies
- controlled random seeds where supported
- additional ansatz families
- explicit noise models
- automated tests for reusable utilities
- comparisons with hardware or runtime primitives
