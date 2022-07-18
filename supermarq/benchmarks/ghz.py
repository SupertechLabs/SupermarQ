import collections

import cirq
from qiskit.quantum_info import hellinger_fidelity

from supermarq.benchmark import Benchmark


class GHZ(Benchmark):
    """Represents the GHZ state preparation benchmark parameterized
    by the number of qubits n.

    Device performance is based on the Hellinger fidelity between
    the experimental and ideal probability distributions.
    """

    def __init__(self, num_qubits: int) -> None:
        """The constructor for the GHZ class.

        Args:
          num_qubits:
            The number of qubits for the circuit.

        Returns:
          A GHZ state with the number of qubits specified.
        """
        self.num_qubits = num_qubits

    def circuit(self) -> cirq.Circuit:
        """Generate an n-qubit GHZ circuit

        Based off the values given to the object constructor.

        Args:
          None.

        Returns:
          The circuit with n-qubits.
        """
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        for i in range(self.num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        circuit.append(cirq.measure(*qubits))
        return circuit

    def score(self, counts: collections.Counter) -> float:
        """Compute the Hellinger fidelity between the experimental and ideal
        results, i.e., 50% probabilty of measuring the all-zero state and 50%
        probability of measuring the all-one state.

        The formula for the Hellinger fidelity between two distributions p and q
        is given by $(sum_i{p_i q_i})^2$.

        Args:
          counts:
            Dictionary of the experimental results. The keys are bitstrings
            represented the measured qubit state, and the values are the number
            of times that state of observed.

        Returns:
          A float representing the score (in this case the Hellinger fidelity).
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.num_qubits: 0.5 for b in ["0", "1"]}
        total_shots = sum(counts.values())
        device_dist = {bitstr: count / total_shots for bitstr, count in counts.items()}
        return hellinger_fidelity(ideal_dist, device_dist)
