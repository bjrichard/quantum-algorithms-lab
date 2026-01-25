"""
vqe_utils

Utilities for running small VQE (Variational Quantum Eigensolver) experiments with:
- exact baseline evaluation (StatevectorEstimator)
- finite-shot evaluation (Aer EstimatorV2)
- repeated runs to quantify variability

Design goals:
- reusable from notebooks and scripts
- minimal side effects (no printing by default)
- returns structured results for analysis and plotting

Acronyms:
- VQE: Variational Quantum Eigensolver
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA


@dataclass(frozen=True)
class VQERunSummary:
    """Summary statistics for repeated VQE runs under identical settings."""
    energies: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.energies))

    @property
    def std(self) -> float:
        return float(np.std(self.energies))

    @property
    def min(self) -> float:
        return float(np.min(self.energies))

    @property
    def max(self) -> float:
        return float(np.max(self.energies))


def make_two_local_ansatz(
    *,
    num_qubits: int,
    reps: int = 1,
    rotation_blocks: str | Sequence[str] = "ry",
    entanglement_blocks: str | Sequence[str] = "cx",
) -> QuantumCircuit:
    """
    Create a small hardware-efficient TwoLocal ansatz and decompose to primitive gates.

    Notes:
    - TwoLocal is deprecated in Qiskit 2.1+ but still works. We keep it for minimal VQE exposure.
    - Decomposition avoids Aer treating the template as an opaque instruction.

    Args:
        num_qubits: Number of qubits in the ansatz circuit.
        reps: Number of repeated layers.
        rotation_blocks: Rotation block(s), e.g., "ry" or ["ry", "rz"].
        entanglement_blocks: Entangling block(s), e.g., "cx".

    Returns:
        A decomposed QuantumCircuit suitable for execution backends.
    """
    ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        reps=reps,
    )
    return ansatz.decompose()


def run_vqe_exact(
    *,
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    maxiter: int = 100,
) -> float:
    """
    Run VQE using exact expectation values via StatevectorEstimator (no sampling noise).

    Args:
        operator: Hamiltonian (observable) to minimize.
        ansatz: Parameterized circuit family.
        maxiter: Optimizer max iterations (COBYLA).

    Returns:
        Estimated minimum eigenvalue (energy).
    """
    from qiskit.primitives import StatevectorEstimator

    estimator = StatevectorEstimator()
    optimizer = COBYLA(maxiter=maxiter)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(operator)
    return float(np.real(result.eigenvalue))


def run_vqe_finite_shots(
    *,
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    optimizer,
    shots: int = 256,
) -> float:
    """
    Run VQE once using Aer EstimatorV2 with a finite shot count.

    Notes:
    - In your Aer version, shots are configured via `estimator.options.shots`.

    Args:
        operator: Hamiltonian (observable) to minimize.
        ansatz: Parameterized circuit family.
        optimizer: A qiskit_algorithms optimizer instance (e.g., COBYLA(...), SPSA(...)).
        shots: Number of shots for sampling expectation values.

    Returns:
        Estimated minimum eigenvalue (energy).
    """
    try:
        from qiskit_aer.primitives import EstimatorV2
    except ImportError as exc:
        raise ImportError(
            "qiskit-aer is required for finite-shot VQE (EstimatorV2). "
            "Install with: conda install -c conda-forge qiskit-aer (or pip install qiskit-aer)."
        ) from exc

    estimator = EstimatorV2()
    estimator.options.shots = shots

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(operator)
    return float(np.real(result.eigenvalue))


def repeat_vqe_finite_shots(
    *,
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    optimizer,
    shots: int = 256,
    num_runs: int = 3,
    verbose: bool = False,
) -> VQERunSummary:
    """
    Repeat finite-shot VQE runs to quantify run-to-run variability.

    Args:
        operator: Hamiltonian (observable) to minimize.
        ansatz: Parameterized circuit family.
        optimizer: Optimizer instance (e.g., COBYLA(...), SPSA(...)).
        shots: Number of shots per expectation evaluation.
        num_runs: Number of repeated VQE runs.
        verbose: If True, prints per-run energies.

    Returns:
        VQERunSummary containing energies and summary stats.
    """
    energies: list[float] = []
    for i in range(num_runs):
        e = run_vqe_finite_shots(
            operator=operator,
            ansatz=ansatz,
            optimizer=optimizer,
            shots=shots,
        )
        energies.append(e)
        if verbose:
            print(f"Run {i+1}/{num_runs}: energy = {e}")

    return VQERunSummary(energies=energies)