"""09_vqe_noise_and_optimization_sensitivity

VQE modifications (noise + optimizer sensitivity)

Purpose:
- Demonstrate how finite-shot sampling makes the VQE objective stochastic.
- Compare optimizer behavior (COBYLA vs SPSA) under identical Hamiltonian/ansatz/shots.
- Contrast against an exact baseline (StatevectorEstimator) for context.

Core idea:
VQE (Variational Quantum Eigensolver) minimizes the energy expectation value
    E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩
over a restricted family of states |ψ(θ)⟩ defined by an ansatz circuit.

Notes:
- The *exact* baseline uses StatevectorEstimator (deterministic, no shot noise).
- The *noisy* experiments use Aer EstimatorV2 with a finite number of shots.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA


# --- 1) Define a simple 2-qubit Hamiltonian H ---
# H = 1.0 * (Z ⊗ Z) + 0.5 * (X ⊗ I) + 0.5 * (I ⊗ X)
H = SparsePauliOp.from_list(
    [
        ("ZZ", 1.0),
        ("XI", 0.5),
        ("IX", 0.5),
    ]
)


def make_ansatz(reps: int = 1):
    """Create a small hardware-efficient ansatz and decompose to primitive gates.

    TwoLocal is convenient for first exposure, but it may appear as an opaque instruction
    to some backends unless decomposed.
    """
    ansatz = TwoLocal(
        num_qubits=2,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        reps=reps,
    )
    return ansatz.decompose()


# --- 2) Baseline: exact expectation values (StatevectorEstimator) ---
def run_exact_baseline(reps: int = 1, maxiter: int = 100) -> float:
    from qiskit.primitives import StatevectorEstimator

    ansatz = make_ansatz(reps=reps)
    estimator = StatevectorEstimator()
    optimizer = COBYLA(maxiter=maxiter)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(H)
    return float(np.real(result.eigenvalue))


# --- 3) Finite-shot experiments: Aer EstimatorV2 ---
def run_vqe_with_shots(
    optimizer, *, reps: int = 1, shots: int = 256, maxiter: int = 100
) -> float:
    """Run VQE once using Aer EstimatorV2 with a finite shot count."""
    from qiskit_aer.primitives import EstimatorV2

    ansatz = make_ansatz(reps=reps)

    estimator = EstimatorV2()
    # In this Aer version, shots are configured via the options object.
    estimator.options.shots = shots

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(H)
    return float(np.real(result.eigenvalue))

def repeat_experiment(
    name: str,
    optimizer,
    *,
    reps: int,
    shots: int,
    maxiter: int,
    num_runs: int,
) -> list[float]:
    """Repeat a VQE configuration multiple times to quantify run-to-run variability."""
    energies: list[float] = []
    print(f"\n--- {name}: reps={reps}, shots={shots}, maxiter={maxiter} ---")
    for i in range(num_runs):
        e = run_vqe_with_shots(optimizer, reps=reps, shots=shots, maxiter=maxiter)
        energies.append(e)
        print(f"Run {i+1}: energy = {e}")
    print(f"Mean ± std: {float(np.mean(energies))} ± {float(np.std(energies))}")
    return energies


def main() -> None:
    # Configuration
    reps = 1
    shots = 256
    maxiter = 100
    num_runs = 3

    # Exact baseline for context (no shot noise)
    e_exact = run_exact_baseline(reps=reps, maxiter=maxiter)
    print("Exact baseline (StatevectorEstimator):")
    print(f"reps={reps}, optimizer=COBYLA, energy={e_exact}")

    # Finite-shot optimizer comparison
    energies_cobyla = repeat_experiment(
        "COBYLA",
        COBYLA(maxiter=maxiter),
        reps=reps,
        shots=shots,
        maxiter=maxiter,
        num_runs=num_runs,
    )
    energies_spsa = repeat_experiment(
        "SPSA",
        SPSA(maxiter=maxiter),
        reps=reps,
        shots=shots,
        maxiter=maxiter,
        num_runs=num_runs,
    )

    print("\n=== Summary (lower is better) ===")
    print(f"Exact baseline energy: {e_exact}")
    print(
        f"COBYLA mean ± std: {float(np.mean(energies_cobyla))} ± {float(np.std(energies_cobyla))}"
    )
    print(
        f"SPSA   mean ± std: {float(np.mean(energies_spsa))} ± {float(np.std(energies_spsa))}"
    )

    print(
        "\nInterpretation:\n"
        "- Exact evaluation is deterministic; optimizer choice matters little once the ansatz is expressive enough.\n"
        "- Finite-shot evaluation introduces stochasticity; repeated runs produce different final energies.\n"
        "- SPSA is designed for noisy objectives, while deterministic optimizers can be more sensitive to fluctuations.\n"
    )


if __name__ == "__main__":
    main()
