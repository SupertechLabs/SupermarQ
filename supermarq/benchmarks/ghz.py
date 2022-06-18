import collections
from typing import Union

import cirq
import qiskit
from qiskit.quantum_info import hellinger_fidelity

import supermarq as sm
from supermarq.benchmark import Benchmark


class GHZ(Benchmark):
    """Represents the GHZ state preparation benchmark parameterized
    by the number of qubits n.

    Device performance is based on the Hellinger fidelity between
    the experimental and ideal probability distributions.
    """

    def __init__(self, num_qubits: int, sdk: str = "cirq") -> None:
        self.num_qubits = num_qubits

        if sdk not in ["cirq", "qiskit"]:
            raise ValueError("Valid sdks are: 'cirq', 'qiskit'")

        self.sdk = sdk

    def circuit(self) -> Union[cirq.Circuit, qiskit.QuantumCircuit]:
        """Generate an n-qubit GHZ circuit"""
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        for i in range(self.num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        circuit.append(cirq.measure(*qubits))

        if self.sdk == "qiskit":
            return sm.converters.cirq_to_qiskit(circuit)

        return circuit

    def score(self, counts: collections.Counter) -> float:
        r"""Compute the Hellinger fidelity between the experimental and ideal
        results, i.e., 50% probabilty of measuring the all-zero state and 50%
        probability of measuring the all-one state.

        The formula for the Hellinger fidelity between two distributions p and q
        is given by $(\sum_i{p_i q_i})^2$.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.num_qubits: 0.5 for b in ["0", "1"]}
        total_shots = sum(counts.values())
        device_dist = {bitstr: count / total_shots for bitstr, count in counts.items()}
        return hellinger_fidelity(ideal_dist, device_dist)
