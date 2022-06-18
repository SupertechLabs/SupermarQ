import abc
from typing import Any, Sequence, Union

import cirq
import qiskit


class Benchmark:
    """Class representing a quantum benchmark application.

    Concrete subclasses must implement the abstract methods ``circuit()`` and
    ``score()``.

    Each instantiation of a Benchmark object represents a single, fully defined
    benchmark application. All the relevant parameters for a benchmark should
    be passed in upon creation, and will be used to generate the correct circuit
    and compute the final score.
    """

    @abc.abstractmethod
    def circuit(
        self,
    ) -> Union[
        cirq.Circuit, Sequence[cirq.Circuit], qiskit.QuantumCircuit, Sequence[qiskit.QuantumCircuit]
    ]:
        """Returns the quantum circuit corresponding to the current benchmark parameters."""

    @abc.abstractmethod
    def score(self, counts: Any) -> float:
        """Returns a normalized [0,1] score reflecting device performance."""
