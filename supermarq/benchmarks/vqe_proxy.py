import collections
import copy
from typing import Counter, List, Tuple, Union

import cirq
import numpy as np
import qiskit
import scipy.optimize as opt

import supermarq as sm


class VQEProxy(sm.benchmark.Benchmark):
    """Proxy benchmark of a full VQE application that targets a single iteration
    of the whole variational optimization.

    The benchmark is parameterized by the number of qubits, n. For each value of
    n, we classically optimize the ansatz, sample 3 iterations near convergence,
    and use the sampled parameters to execute the corresponding circuits on the
    QPU. We take the measured energies from these experiments and average their
    values and compute a score based on how closely the experimental results are
    to the noiseless values.
    """

    def __init__(self, num_qubits: int, num_layers: int = 1, sdk: str = "cirq") -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.hamiltonian = self._gen_tfim_hamiltonian()
        self._params = self._gen_angles()

        if sdk not in ["cirq", "qiskit"]:
            raise ValueError("Valid sdks are: 'cirq', 'qiskit'")

        self.sdk = sdk

    def _gen_tfim_hamiltonian(self) -> List:
        r"""Generate an n-qubit Hamiltonian for a transverse-field Ising model (TFIM).

            $H = \sum_i^n(X_i) + \sum_i^n(Z_i Z_{i+1})$

        Example of a 6-qubit TFIM Hamiltonian:

            $H_6 = XIIIII + IXIIII + IIXIII + IIIXII + IIIIXI + IIIIIX + ZZIIII
                  + IZZIII + IIZZII + IIIZZI + IIIIZZ + ZIIIIZ$
        """
        hamiltonian = []
        for i in range(self.num_qubits):
            hamiltonian.append(["X", i, 1])  # [Pauli type, qubit idx, weight]
        for i in range(self.num_qubits - 1):
            hamiltonian.append(["ZZ", (i, i + 1), 1])
        hamiltonian.append(["ZZ", (self.num_qubits - 1, 0), 1])
        return hamiltonian

    def _gen_ansatz(self, params: List[float]) -> List[cirq.Circuit]:
        qubits = cirq.LineQubit.range(self.num_qubits)
        z_circuit = cirq.Circuit()

        param_counter = 0
        for _ in range(self.num_layers):
            # Ry rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Ry(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1
            # Rz rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Rz(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1
            # Entanglement block
            for i in range(self.num_qubits - 1):
                z_circuit.append(cirq.CX(qubits[i], qubits[i + 1]))
            # Ry rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Ry(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1
            # Rz rotation block
            for i in range(self.num_qubits):
                z_circuit.append(cirq.Rz(rads=2 * params[param_counter])(qubits[i]))
                param_counter += 1

        x_circuit = copy.deepcopy(z_circuit)
        x_circuit.append(cirq.H(q) for q in qubits)

        # Measure all qubits
        z_circuit.append(cirq.measure(*qubits))
        x_circuit.append(cirq.measure(*qubits))

        return [z_circuit, x_circuit]

    def _parity_ones(self, bitstr: str) -> int:
        one_count = 0
        for i in bitstr:
            if i == "1":
                one_count += 1
        return one_count % 2

    def _calc(self, bit_list: List[str], bitstr: str, probs: Counter) -> float:
        energy = 0.0
        for item in bit_list:
            if self._parity_ones(item) == 0:
                energy += probs.get(bitstr, 0)
            else:
                energy -= probs.get(bitstr, 0)
        return energy

    def _get_expectation_value_from_probs(self, probs_z: Counter, probs_x: Counter) -> float:
        avg_energy = 0.0

        # Find the contribution to the energy from the X-terms: \sum_i{X_i}
        for bitstr in probs_x.keys():
            bit_list_x = [bitstr[i] for i in range(len(bitstr))]
            avg_energy += self._calc(bit_list_x, bitstr, probs_x)

        # Find the contribution to the energy from the Z-terms: \sum_i{Z_i Z_{i+1}}
        for bitstr in probs_z.keys():
            # fmt: off
            bit_list_z = [bitstr[(i - 1): (i + 1)] for i in range(1, len(bitstr))]
            # fmt: on
            bit_list_z.append(bitstr[0] + bitstr[-1])  # Add the wrap-around term manually
            avg_energy += self._calc(bit_list_z, bitstr, probs_z)

        return avg_energy

    def _get_opt_angles(self) -> Tuple[List, float]:
        def f(params: List) -> float:
            z_circuit, x_circuit = self._gen_ansatz(params)
            z_probs = sm.simulation.get_ideal_counts(z_circuit)
            x_probs = sm.simulation.get_ideal_counts(x_circuit)
            energy = self._get_expectation_value_from_probs(z_probs, x_probs)

            return -energy  # because we are minimizing instead of maximizing

        init_params = [
            np.random.uniform() * 2 * np.pi for _ in range(self.num_layers * 4 * self.num_qubits)
        ]
        out = opt.minimize(f, init_params, method="COBYLA")

        return out["x"], out["fun"]

    def _gen_angles(self) -> List:
        """Classically simulate the variational optimization and return
        the final parameters.
        """
        params, _ = self._get_opt_angles()
        return params

    def circuit(self) -> Union[List[cirq.Circuit], List[qiskit.QuantumCircuit]]:
        """Construct a parameterized ansatz.

        Returns a list of circuits: the ansatz measured in the Z basis, and the
        ansatz measured in the X basis. The counts obtained from evaluated these
        two circuits should be passed to `score` in the same order they are
        returned here.
        """
        circuits = self._gen_ansatz(self._params)

        if self.sdk == "qiskit":
            return [sm.converters.cirq_to_qiskit(circuit) for circuit in circuits]

        return circuits

    def score(self, counts: List[Counter]) -> float:
        """Compare the average energy measured by the experiments to the ideal
        value obtained via noiseless simulation. In principle the ideal value
        can be obtained through efficient classical means since the 1D TFIM
        is analytically solvable.
        """
        counts_z, counts_x = counts
        shots_z = sum(counts_z.values())
        probs_z = {bitstr: count / shots_z for bitstr, count in counts_z.items()}
        shots_x = sum(counts_x.values())
        probs_x = {bitstr: count / shots_x for bitstr, count in counts_x.items()}
        experimental_expectation = self._get_expectation_value_from_probs(
            collections.Counter(probs_z),
            collections.Counter(probs_x),
        )

        circuit_z, circuit_x = self.circuit()
        ideal_expectation = self._get_expectation_value_from_probs(
            sm.simulation.get_ideal_counts(circuit_z),
            sm.simulation.get_ideal_counts(circuit_x),
        )

        return float(
            1.0 - abs(ideal_expectation - experimental_expectation) / abs(2 * ideal_expectation)
        )
