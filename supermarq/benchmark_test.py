import collections

import cirq
import pytest
from qiskit.quantum_info import hellinger_fidelity

from supermarq.benchmark import Benchmark


@pytest.fixture
def benchmark() -> Benchmark:
    """Simple one-qubit benchmark that creates an equal superposition state"""

    class _TestBenchmark(Benchmark):
        def circuit(self) -> cirq.Circuit:
            qubit = cirq.LineQubit(0)
            return cirq.Circuit(cirq.H(qubit), cirq.measure(qubit))

        def score(self, counts: collections.Counter) -> float:
            dist = {b: c / sum(counts.values()) for b, c in counts.items()}
            return hellinger_fidelity({"0": 0.5, "1": 0.5}, dist)

    return _TestBenchmark()


def test_benchmark_circuit(benchmark: Benchmark) -> None:
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)), cirq.measure(cirq.LineQubit(0)))
    assert benchmark.circuit() == circuit


def test_benchmark_score(benchmark: Benchmark) -> None:
    assert benchmark.score(collections.Counter({"0": 50, "1": 50})) == 1
