import pytest
import qiskit

import supermarq
from supermarq.benchmarks.vqe_proxy import VQEProxy


def test_vqe_circuit() -> None:
    vqe = VQEProxy(3, 1)
    assert len(vqe.circuit()) == 2
    assert len(vqe.circuit()[0].all_qubits()) == 3

    vqe = VQEProxy(3, 1, sdk="qiskit")
    assert isinstance(vqe.circuit()[0], qiskit.QuantumCircuit)

    with pytest.raises(ValueError):
        VQEProxy(3, 1, sdk="")


def test_vqe_score() -> None:
    vqe = VQEProxy(3, 1)
    circuits = vqe.circuit()
    probs = [supermarq.simulation.get_ideal_counts(circ) for circ in circuits]
    assert vqe.score(probs) > 0.99
