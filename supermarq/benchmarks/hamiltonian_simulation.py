import collections
from typing import Union

import cirq
import numpy as np
import qiskit

import supermarq as sm
from supermarq.benchmark import Benchmark


class HamiltonianSimulation(Benchmark):
    """Quantum benchmark focused on the ability to simulate 1D
    Transverse Field Ising Models (TFIM) of variable length.

    Device performance is based on how closely the experimentally obtained
    average magnetization (along the Z-axis) matches the noiseless value.
    Since the 1D TFIM is efficiently simulatable with classical algorithms,
    computing the noiseless average magnetization remains scalable over a large
    range of benchmark sizes.
    """

    def __init__(
        self, num_qubits: int, time_step: int = 1, total_time: int = 1, sdk: str = "cirq"
    ) -> None:
        """Args:
        num_qubits: int
            Size of the TFIM chain, equivalent to the number of qubits.
        time_step: int
            Size of the timestep in attoseconds.
        total_time:
            Total simulation time of the TFIM chain in attoseconds.
        """
        self.num_qubits = num_qubits
        self.time_step = time_step
        self.total_time = total_time

        if sdk not in ["cirq", "qiskit"]:
            raise ValueError("Valid sdks are: 'cirq', 'qiskit'")

        self.sdk = sdk

    def circuit(self) -> Union[cirq.Circuit, qiskit.QuantumCircuit]:
        """Generate a circuit to simulate the evolution of an n-qubit TFIM
        chain under the Hamiltonian:

        H(t) = - Jz * sum_{i=1}^{n-1}(sigma_{z}^{i} * sigma_{z}^{i+1})
               - e_ph * cos(w_ph * t) * sum_{i=1}^{n}(sigma_{x}^{i})

        where,
            w_ph: frequency of E" phonon in MoSe2.
            e_ph: strength of electron-phonon coupling.
        """
        hbar = 0.658212  # eV*fs
        jz = (
            hbar * np.pi / 4
        )  # eV, coupling coeff; Jz<0 is antiferromagnetic, Jz>0 is ferromagnetic
        freq = 0.0048  # 1/fs, frequency of MoSe2 phonon

        w_ph = 2 * np.pi * freq
        e_ph = 3 * np.pi * hbar / (8 * np.cos(np.pi * freq))

        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()

        # Build up the circuit over total_time / time_step propagation steps
        for step in range(int(self.total_time / self.time_step)):
            # Simulate the Hamiltonian term-by-term
            t = (step + 0.5) * self.time_step

            # Single qubit terms
            psi = -2.0 * e_ph * np.cos(w_ph * t) * self.time_step / hbar
            for qubit in qubits:
                circuit.append(cirq.H(qubit))
                circuit.append(cirq.rz(psi)(qubit))
                circuit.append(cirq.H(qubit))

            # Coupling terms
            psi2 = -2.0 * jz * self.time_step / hbar
            for i in range(len(qubits) - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
                circuit.append(cirq.rz(psi2)(qubits[i + 1]))
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

        # End the circuit with measurements of every qubit in the Z-basis
        circuit.append(cirq.measure(*qubits))

        if self.sdk == "qiskit":
            return sm.converters.cirq_to_qiskit(circuit)

        return circuit

    def _average_magnetization(self, result: dict, shots: int) -> float:
        mag = 0
        for spin_str, count in result.items():
            spin_int = [1 - 2 * int(s) for s in spin_str]
            mag += (
                sum(spin_int) / len(spin_int)
            ) * count  # <Z> weighted by number of times we saw this bitstring
        average_mag = mag / shots  # normalize by the total number of shots
        return average_mag

    def score(self, counts: collections.Counter) -> float:
        """Compute the average magnetization of the TFIM chain along the Z-axis
        for the experimental results and via noiseless simulation.

        Args:
            counts: Dictionary of the experimental results. The keys are bitstrings
                represented the measured qubit state, and the values are the number
                of times that state of observed.
        """
        ideal_counts = sm.simulation.get_ideal_counts(self.circuit())

        total_shots = sum(counts.values())

        mag_ideal = self._average_magnetization(ideal_counts, 1)
        mag_experimental = self._average_magnetization(counts, total_shots)

        return 1 - abs(mag_ideal - mag_experimental) / 2
