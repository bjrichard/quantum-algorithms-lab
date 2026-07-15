"""Utilities for small Variational Quantum Eigensolver experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA


@dataclass(frozen=True)
class VQERunSummary:
    """Store energies and summary statistics for repeated VQE runs."""

    energies: list[float]

    def __post_init__(self) -> None:
        if not self.energies:
            raise ValueError("energies must contain at least one value.")

    @property
    def mean(self) -> float:
        """Return the arithmetic mean of the observed energies."""
        return float(np.mean(self.energies))

    @property
    def std(self) -> float:
        """Return the population standard deviation of the observed energies."""
        return float(np.std(self.energies))

    @property
    def min(self) -> float:
        """Return the smallest observed energy."""
        return float(np.min(self.energies))

    @property
    def max(self) -> float:
        """Return the largest observed energy."""
        return float(np.max(self.energies))


def make_two_local_ansatz(
    *,
    num_qubits: int,
    reps: int = 1,
    rotation_blocks: str | Sequence[str] = "ry",
    entanglement_blocks: str | Sequence[str] = "cx",
) -> QuantumCircuit:
    """Create a small hardware-efficient N-local ansatz.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Number of repeated rotation-and-entanglement layers.
    rotation_blocks
        Rotation block or blocks used in each layer.
    entanglement_blocks
        Entangling block or blocks used in each layer.

    Returns
    -------
    QuantumCircuit
        A decomposed parameterized circuit suitable for VQE.

    Raises
    ------
    ValueError
        If ``num_qubits`` or ``reps`` is not positive.

    Notes
    -----
    Qiskit's function-based ``n_local`` constructor replaces the deprecated
    ``TwoLocal`` circuit class.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least one.")
    if reps < 1:
        raise ValueError("reps must be at least one.")

    return n_local(
        num_qubits=num_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        reps=reps,
    ).decompose()


def run_vqe_exact(
    *,
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    maxiter: int = 100,
) -> float:
    """Run VQE using exact statevector expectation values."""
    if maxiter < 1:
        raise ValueError("maxiter must be at least one.")

    from qiskit.primitives import StatevectorEstimator

    vqe = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
    )
    result = vqe.compute_minimum_eigenvalue(operator)
    return float(np.real(result.eigenvalue))


def run_vqe_finite_shots(
    *,
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    optimizer: Any,
    shots: int = 256,
) -> float:
    """Run one finite-shot VQE experiment using Qiskit Aer."""
    if shots < 1:
        raise ValueError("shots must be at least one.")

    try:
        from qiskit_aer.primitives import EstimatorV2
    except ImportError as exc:
        raise ImportError(
            "qiskit-aer is required for finite-shot VQE. "
            "Install the repository environment from environment.yml."
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
    optimizer: Any,
    shots: int = 256,
    num_runs: int = 3,
    verbose: bool = False,
) -> VQERunSummary:
    """Repeat finite-shot VQE and summarize the observed energies."""
    if num_runs < 1:
        raise ValueError("num_runs must be at least one.")
    if shots < 1:
        raise ValueError("shots must be at least one.")

    energies: list[float] = []
    for run_index in range(num_runs):
        energy = run_vqe_finite_shots(
            operator=operator,
            ansatz=ansatz,
            optimizer=optimizer,
            shots=shots,
        )
        energies.append(energy)
        if verbose:
            print(f"Run {run_index + 1}/{num_runs}: energy = {energy}")

    return VQERunSummary(energies=energies)
