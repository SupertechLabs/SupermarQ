import collections
from typing import Generator, List, Union

import cirq
import qiskit
from qiskit.quantum_info import hellinger_fidelity

import supermarq as sm
from supermarq.benchmark import Benchmark


class PhaseCode(Benchmark):
    """
    Creates a circuit for syndrome measurement in a phase-flip error
    correcting code.

    Args:
    - num_data: The number of data qubits
    - num_rounds: The number of measurement rounds
    - phase_state: A list of zeros and ones denoting the state to initialize each data
                   qubit to. Currently just + or - states. 0 -> +, 1 -> -

    returns a cirq circuit for the phase-flip error correcting code

    """

    def __init__(
        self, num_data_qubits: int, num_rounds: int, phase_state: List[int], sdk: str = "cirq"
    ) -> None:
        if len(phase_state) != num_data_qubits:
            raise ValueError("The length of `phase_state` must match the number of data qubits")
        self.num_data_qubits = num_data_qubits
        self.num_rounds = num_rounds
        self.phase_state = phase_state

        if sdk not in ["cirq", "qiskit"]:
            raise ValueError("Valid sdks are: 'cirq', 'qiskit'")

        self.sdk = sdk

    def _measurement_round_cirq(self, qubits: List[cirq.LineQubit], round_idx: int) -> Generator:
        """
        Generates cirq ops for a single measurement round

        Args:
        - qubits: Circuit qubits - assumed data on even indices and
                  measurement on odd indices
        """
        ancilla_qubits = []
        yield [cirq.H(q) for q in qubits]
        for i in range(1, len(qubits), 2):
            yield cirq.CZ(qubits[i - 1], qubits[i])
            yield cirq.CZ(qubits[i + 1], qubits[i])
            ancilla_qubits.append(qubits[i])
        yield [cirq.H(q) for q in qubits]
        yield cirq.measure(*ancilla_qubits, key=f"mcm{round_idx}")
        yield [cirq.ops.reset(qubit) for qubit in ancilla_qubits]

    def circuit(self) -> Union[cirq.Circuit, qiskit.QuantumCircuit]:
        num_qubits = 2 * self.num_data_qubits - 1
        qubits = cirq.LineQubit.range(num_qubits)
        circuit = cirq.Circuit()

        # Initialize the data qubits
        for i in range(self.num_data_qubits):
            if self.phase_state[i] == 1:
                circuit.append(cirq.X(qubits[2 * i]))
            circuit.append(cirq.H(qubits[2 * i]))

        # Apply measurement rounds
        circuit.append(self._measurement_round_cirq(qubits, i) for i in range(self.num_rounds))

        # Measure final outcomes in X basis to produce single product state
        for i in range(self.num_data_qubits):
            circuit.append(cirq.H(qubits[2 * i]))

        circuit.append(cirq.measure(*qubits, key="meas_all"))

        if self.sdk == "qiskit":
            return sm.converters.cirq_to_qiskit(circuit)

        return circuit

    def _get_ideal_dist(self) -> collections.Counter:
        """Return the ideal probability distribution of self.circuit().

        Since the initial states of the data qubits are either |+> or |->,
        and we measure the final state in the X-basis, the final state is a
        single product state in the noiseless case.
        """
        ancilla_state, final_state = "", ""
        for i in range(self.num_data_qubits - 1):
            # parity checks
            ancilla_state += str((self.phase_state[i] + self.phase_state[i + 1]) % 2)
            final_state += str(self.phase_state[i]) + "0"
        else:
            final_state += str(self.phase_state[-1])

        ideal_bitstring = [ancilla_state] * self.num_rounds + [final_state]
        return collections.Counter({" ".join(ideal_bitstring): 1.0})

    def score(self, counts: collections.Counter) -> float:
        """Device performance is given by the Hellinger fidelity between
        the experimental results and the ideal distribution. The ideal
        is known based on the phase_state parameter.
        """
        ideal_dist = self._get_ideal_dist()
        total_shots = sum(counts.values())
        experimental_dist = {bitstr: shots / total_shots for bitstr, shots in counts.items()}
        return hellinger_fidelity(ideal_dist, experimental_dist)
