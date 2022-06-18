import collections

import pytest
import qiskit

from supermarq.benchmarks.phase_code import PhaseCode


def test_phase_code_circuit() -> None:
    pc = PhaseCode(3, 1, [1, 1, 1])
    assert len(pc.circuit().all_qubits()) == 5

    pc = PhaseCode(3, 1, [1, 1, 1], sdk="qiskit")
    assert isinstance(pc.circuit(), qiskit.QuantumCircuit)

    with pytest.raises(ValueError):
        PhaseCode(3, 1, [1, 1, 1], sdk="")


def test_phase_code_score() -> None:
    pc = PhaseCode(4, 2, [0, 1, 1, 0])
    assert pc.score(collections.Counter({"101 101 0010100": 100})) == 1


def test_invalid_size() -> None:
    with pytest.raises(
        ValueError, match="The length of `phase_state` must match the number of data qubits"
    ):
        PhaseCode(3, 1, [0])
