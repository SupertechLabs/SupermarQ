"""Definition of the Fermionic SWAP QAOA benchmark within the SupermarQ suite."""
import collections
from typing import List, Tuple, Union

import cirq
import numpy as np
import qiskit
import scipy

import supermarq as sm
from supermarq.benchmark import Benchmark


class QAOAFermionicSwapProxy(Benchmark):
    """Proxy of a full Quantum Approximate Optimization Algorithm (QAOA) benchmark.

    This benchmark targets MaxCut on a Sherrington-Kirkpatrick (SK) model. Device
    performance is given by the Hellinger fidelity between the experimental output
    distribution and the true distribution obtained via scalable, classical simulation.

    The ansatz for this QAOA problem utilizes the fermionic SWAP network which is able
    to perform all of the required O(N^2) interactions in linear circuit depth. This
    ansatz is especially well-suited to QPU architectures which only support
    nearest-neighbor connectivity. See https://doi.org/10.3390/electronics10141690 for
    an example of this ansatz used in practice.

    When a new instance of this benchmark is created, the ansatz parameters will
    be initialized by:
        1. Generating a random instance of an SK graph
        2. Finding approximately optimal angles (rather than random values)
    """

    def __init__(self, num_qubits: int, sdk: str = "cirq") -> None:
        """Generate a new benchmark instance.

        Args:
            num_qubits: The number of nodes (qubits) within the SK graph.
        """
        self.num_qubits = num_qubits
        self.hamiltonian = self._gen_sk_hamiltonian()
        self.params = self._gen_angles()

        if sdk not in ["cirq", "qiskit"]:
            raise ValueError("Valid sdks are: 'cirq', 'qiskit'")

        self.sdk = sdk

    def _gen_sk_hamiltonian(self) -> List:
        """Randomly pick +1 or -1 for each edge weight."""
        hamiltonian = []
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                hamiltonian.append([i, j, np.random.choice([-1, 1])])

        np.random.shuffle(hamiltonian)

        return hamiltonian

    def _gen_swap_network(self, gamma: float, beta: float) -> cirq.Circuit:
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()

        # initialize |++++>
        for q in qubits:
            circuit.append(cirq.H(q))

        # Implement the phase-separator unitary with a swap network
        # The covers indicate which qubits will be swapped at each layer
        cover_a = [(idx - 1, idx) for idx in range(1, self.num_qubits, 2)]
        cover_b = [(idx - 1, idx) for idx in range(2, self.num_qubits, 2)]

        # The indices of the virtual map correspond to physical qubits,
        # the value at that index corresponds to the virtual qubit residing there
        virtual_map = np.arange(self.num_qubits)

        for layer in range(self.num_qubits):
            cover = [cover_a, cover_b][layer % 2]
            for pair in cover:
                i, j = pair  # swap physical qubits i and j

                # Get the corresponding weight between the virtual qubits
                v_i = virtual_map[i]
                v_j = virtual_map[j]
                for edge in self.hamiltonian:
                    if v_i == edge[0] and v_j == edge[1]:
                        weight = edge[2]
                phi = gamma * weight

                # Perform the ZZ+SWAP operation
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))
                circuit.append(cirq.rz(2 * phi)(qubits[j]))
                circuit.append(cirq.CNOT(qubits[j], qubits[i]))
                circuit.append(cirq.CNOT(qubits[i], qubits[j]))

                # update the virtual map
                virtual_map[j], virtual_map[i] = virtual_map[i], virtual_map[j]

        # Implement the mixing unitary
        for q in qubits:
            circuit.append(cirq.rx(2 * beta)(q))

        # Measure all qubits
        circuit.append(cirq.measure(*qubits))

        # NOTE: the final qubits in this circuit are in REVERSED order due to the swap network
        return circuit

    def _get_energy_for_bitstring(self, bitstring: str) -> float:
        energy_val = 0
        for i, j, weight in self.hamiltonian:
            if bitstring[i] == bitstring[j]:
                energy_val -= weight  # if edge is UNCUT, weight counts against objective
            else:
                energy_val += weight  # if edge is CUT, weight counts towards objective
        return energy_val

    def _get_expectation_value_from_probs(self, probabilities: collections.Counter) -> float:
        expectation_value = 0.0
        for bitstring, probability in probabilities.items():
            expectation_value += probability * self._get_energy_for_bitstring(bitstring)
        return expectation_value

    def _get_opt_angles(self) -> Tuple[List, float]:
        def f(params: List) -> float:
            gamma, beta = params
            circ = self._gen_swap_network(gamma, beta)
            # Reverse bitstring ordering due to SWAP network
            raw_probs = sm.simulation.get_ideal_counts(circ)
            probs = collections.Counter(
                {bitstring[::-1]: probability for bitstring, probability in raw_probs.items()}
            )
            h_expect = self._get_expectation_value_from_probs(probs)

            return -h_expect  # because we are minimizing instead of maximizing

        init_params = [np.random.uniform() * 2 * np.pi, np.random.uniform() * 2 * np.pi]
        out = scipy.optimize.minimize(f, init_params, method="COBYLA")

        return out["x"], out["fun"]

    def _gen_angles(self) -> List:
        # Classically simulate the variational optimization 5 times,
        # return the parameters from the best performing simulation
        best_params, best_cost = [], 0.0
        for _ in range(5):
            params, cost = self._get_opt_angles()
            if cost < best_cost:
                best_params = params
                best_cost = cost
        return best_params

    def circuit(self) -> Union[cirq.Circuit, qiskit.QuantumCircuit]:
        """Generate a QAOA circuit for the Sherrington-Kirkpatrick model.

        This particular benchmark utilizes a quantum circuit structure called
        the fermionic swap network. We restrict the depth of this proxy benchmark
        to p=1 to keep the classical simulation scalable.
        """
        gamma, beta = self.params
        circuit = self._gen_swap_network(gamma, beta)

        if self.sdk == "qiskit":
            return sm.converters.cirq_to_qiskit(circuit)

        return circuit

    def score(self, counts: collections.Counter) -> float:
        """Compare the experimental output to the output of noiseless simulation.

        The implementation here has exponential runtime and would not scale.
        However, it could in principle be done efficiently via
        https://arxiv.org/abs/1706.02998, so we're good.
        """
        # Reverse bitstring ordering due to SWAP network
        raw_probs = sm.simulation.get_ideal_counts(self.circuit())
        ideal_counts = collections.Counter(
            {bitstring[::-1]: probability for bitstring, probability in raw_probs.items()}
        )
        total_shots = sum(counts.values())
        # Reverse the order of the bitstrings due to the fermionic swap ansatz
        experimental_counts = collections.Counter(
            {k[::-1]: v / total_shots for k, v in counts.items()}
        )

        ideal_value = self._get_expectation_value_from_probs(ideal_counts)
        experimental_value = self._get_expectation_value_from_probs(experimental_counts)

        return 1 - abs(ideal_value - experimental_value) / (2 * ideal_value)
