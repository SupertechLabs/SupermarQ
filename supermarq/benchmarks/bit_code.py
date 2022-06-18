import collections
from typing import Generator, List, Union

import cirq
import qiskit
from qiskit.quantum_info import hellinger_fidelity

import supermarq as sm
from supermarq.benchmark import Benchmark


class BitCode(Benchmark):
    """Creates a circuit for syndrome measurement in a bit-flip error correcting code.

    Args:
    - num_data: The number of data qubits
    - num_rounds: The number of measurement rounds
    - bit_state: A list denoting the state to initialize each data qubit to.

    returns a cirq circuit for the bit-flip error correcting code
    """

    def __init__(
        self, num_data_qubits: int, num_rounds: int, bit_state: List[int], sdk: str = "cirq"
    ) -> None:
        if len(bit_state) != num_data_qubits:
            raise ValueError("The length of `bit_state` must match the number of data qubits")
        self.num_data_qubits = num_data_qubits
        self.num_rounds = num_rounds
        self.bit_state = bit_state

        if sdk not in ["cirq", "qiskit"]:
            raise ValueError("Valid sdks are: 'cirq', 'qiskit'")

        self.sdk = sdk

    def _measurement_round_cirq(self, qubits: List[cirq.LineQubit], round_idx: int) -> Generator:
        """
        Generates cirq ops for a single measurement round

        Args:
        - qubits: Circuit qubits - assumed data on even indices and measurement on odd indices
        """
        ancilla_qubits = []
        for i in range(1, len(qubits), 2):
            yield cirq.CX(qubits[i - 1], qubits[i])
            yield cirq.CX(qubits[i + 1], qubits[i])
            ancilla_qubits.append(qubits[i])
        yield cirq.measure(*ancilla_qubits, key=f"mcm{round_idx}")
        yield [cirq.ops.reset(qubit) for qubit in ancilla_qubits]

    def circuit(self) -> Union[cirq.Circuit, qiskit.QuantumCircuit]:
        num_qubits = 2 * self.num_data_qubits - 1
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()

        # Initialize the data qubits
        for i in range(self.num_data_qubits):
            if self.bit_state[i] == 1:
                circuit.append(cirq.X(qubits[2 * i]))

        # Apply measurement rounds
        circuit.append(self._measurement_round_cirq(qubits, i) for i in range(self.num_rounds))

        circuit.append(cirq.measure(*qubits, key="meas_all"))

        if self.sdk == "qiskit":
            return sm.converters.cirq_to_qiskit(circuit)

        return circuit

    def _get_ideal_dist(self) -> collections.Counter:
        """Return the ideal probability distribution of self.circuit().

        Since the only allowed initial states for this benchmark are
        single product states, there is a single bitstring that should be
        measured in the noiseless case.
        """
        ancilla_state, final_state = "", ""
        for i in range(self.num_data_qubits - 1):
            # parity checks
            ancilla_state += str((self.bit_state[i] + self.bit_state[i + 1]) % 2)
            final_state += str(self.bit_state[i]) + "0"
        else:
            final_state += str(self.bit_state[-1])

        ideal_bitstring = [ancilla_state] * self.num_rounds + [final_state]
        return collections.Counter({" ".join(ideal_bitstring): 1.0})

    def score(self, counts: collections.Counter) -> float:
        """Device performance is given by the Hellinger fidelity between
        the experimental results and the ideal distribution. The ideal
        is known based on the bit_state parameter.
        """
        ideal_dist = self._get_ideal_dist()
        total_shots = sum(counts.values())
        experimental_dist = {bitstr: shots / total_shots for bitstr, shots in counts.items()}
        return hellinger_fidelity(ideal_dist, experimental_dist)
