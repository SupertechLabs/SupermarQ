import collections

import pytest
import qiskit

from supermarq.benchmarks.ghz import GHZ


def test_ghz_circuit() -> None:
    ghz = GHZ(3)
    assert len(ghz.circuit().all_qubits()) == 3

    ghz = GHZ(3, sdk="qiskit")
    assert isinstance(ghz.circuit(), qiskit.QuantumCircuit)

    with pytest.raises(ValueError):
        GHZ(3, sdk="")


def test_ghz_score() -> None:
    ghz = GHZ(3)
    assert ghz.score(collections.Counter({"000": 500, "111": 500})) == 1
