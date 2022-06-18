import pytest
import qiskit

import supermarq
from supermarq.benchmarks.hamiltonian_simulation import HamiltonianSimulation


def test_hamiltonian_simulation_circuit() -> None:
    hs = HamiltonianSimulation(4, 1, 1)
    assert len(hs.circuit().all_qubits()) == 4

    hs = HamiltonianSimulation(4, 1, 1, sdk="qiskit")
    assert isinstance(hs.circuit(), qiskit.QuantumCircuit)

    with pytest.raises(ValueError):
        HamiltonianSimulation(4, 1, 1, sdk="")


def test_hamiltonian_simulation_score() -> None:
    hs = HamiltonianSimulation(4, 1, 1)
    assert hs._average_magnetization({"1111": 1}, 1) == -1.0
    assert hs._average_magnetization({"0000": 1}, 1) == 1.0
    assert hs.score(supermarq.simulation.get_ideal_counts(hs.circuit())) > 0.99
