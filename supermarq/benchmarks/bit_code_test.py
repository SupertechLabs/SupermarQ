import collections

import pytest
import qiskit

from supermarq.benchmarks.bit_code import BitCode


def test_bit_code_circuit() -> None:
    bc = BitCode(3, 1, [1, 1, 1])
    assert len(bc.circuit().all_qubits()) == 5

    bc = BitCode(3, 1, [1, 1, 1], sdk="qiskit")
    assert isinstance(bc.circuit(), qiskit.QuantumCircuit)

    with pytest.raises(ValueError):
        BitCode(3, 1, [1, 1, 1], sdk="")


def test_bit_code_score() -> None:
    bc = BitCode(4, 2, [0, 1, 1, 0])
    assert bc.score(collections.Counter({"101 101 0010100": 100})) == 1


def test_invalid_size() -> None:
    with pytest.raises(
        ValueError, match="The length of `bit_state` must match the number of data qubits"
    ):
        BitCode(3, 1, [0])
