import collections
from typing import cast, Dict, List, Optional, Tuple

import cirq
import numpy as np
import sympy

from supermarq import stabilizers
from supermarq.benchmark import Benchmark


class MerminBell(Benchmark):
    """The Mermin-Bell benchmark is a test of a quantum computer's ability
    to exploit purely quantum phenomemna such as superposition and entanglement.
    It is based on the famous Bell-inequality tests of locality.

    Performance is based on a QPU's ability to prepare a GHZ state and measure
    the Mermin operator.
    """

    def __init__(self, num_qubits: int) -> None:
        """The constructor for the Mermin-Bell benchmark test.

        Args:
          num_qubits:
            An int representing the size of the TFIM chain, equivalent to the number of qubits.

        Returns:
          An object representing the values for the simulation.
        """

        self.num_qubits = num_qubits
        self.qubits = cirq.LineQubit.range(self.num_qubits)

        self.mermin_operator = self._mermin_operator(self.num_qubits)
        self.stabilizer, self.pauli_basis = stabilizers.construct_stabilizer(
            self.num_qubits, self.mermin_operator
        )

    def circuit(self) -> cirq.Circuit:
        """Generate an n-qubit GHZ circuit.

        Based off the values given to the object constructor.

        Args:
          None.

        Returns:
          The circuit with n-qubits.
        """
        circuit = cirq.Circuit()

        # Create a GHZ state
        circuit.append(cirq.rx(-np.pi / 2).on(self.qubits[0]))
        for i in range(self.num_qubits - 1):
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))

        # Simultaneously measure all terms in the Mermin operator
        measurement_circuit = self._get_measurement_circuit()
        circuit.append(measurement_circuit.get_circuit())

        return circuit

    def score(self, counts: collections.Counter) -> float:
        """Compute the score for the N-qubit Mermin-Bell benchmark.

        This function assumes the regular big endian ordering of bitstring results

        Args:
          counts:
            Dictionary of the experimental results. The keys are bitstrings
            represented the measured qubit state, and the values are the number
            of times that state of observed.

        Returns:
          The score of the simulation (this case the Mermin-Bell benchmark).
        """

        # Store the conjugation rules for H, S, CX, CZ, SWAP in dictionaries. The keys are
        # the pauli strings to be conjugated and the values are the resulting pauli strings
        # after conjugation.
        # The typing here was added to satisfy mypy. Declaring this dict without the explicit
        # typing gets created as Dict[EigenGate, Dict[str, str]], but iterating through a
        # cirq.Circuit and passing op.gate as the key yields type Optional[Gate].
        conjugation_rules: Dict[Optional[cirq.Gate], Dict[str, str]] = {
            cirq.ops.H: {"I": "I", "X": "Z", "Y": "-Y", "Z": "X"},
            cirq.ops.S: {"I": "I", "X": "Y", "Y": "-X", "Z": "Z"},
            cirq.ops.CNOT: {
                "II": "II",
                "IX": "IX",
                "XI": "XX",
                "XX": "XI",
                "IY": "ZY",
                "YI": "YX",
                "YY": "-XZ",
                "IZ": "ZZ",
                "ZI": "ZI",
                "ZZ": "IZ",
                "XY": "YZ",
                "YX": "YI",
                "XZ": "-YY",
                "ZX": "ZX",
                "YZ": "XY",
                "ZY": "IY",
            },
            cirq.ops.CZ: {
                "II": "II",
                "IX": "ZX",
                "XI": "XZ",
                "XX": "YY",
                "IY": "ZY",
                "YI": "YZ",
                "YY": "XX",
                "IZ": "IZ",
                "ZI": "ZI",
                "ZZ": "ZZ",
                "XY": "-YX",
                "YX": "-XY",
                "XZ": "XI",
                "ZX": "IX",
                "YZ": "YI",
                "ZY": "IY",
            },
            cirq.ops.SWAP: {
                "II": "II",
                "IX": "XI",
                "XI": "IX",
                "XX": "XX",
                "IY": "YI",
                "YI": "IY",
                "YY": "YY",
                "IZ": "ZI",
                "ZI": "IZ",
                "ZZ": "ZZ",
                "XY": "YX",
                "YX": "XY",
                "XZ": "ZX",
                "ZX": "XZ",
                "YZ": "ZY",
                "ZY": "YZ",
            },
        }

        measurement_circuit = self._get_measurement_circuit().get_circuit()

        expect_val = 0.0
        for mermin_coef, mermin_pauli in self.mermin_operator:
            # Iterate through the operations in the measurement circuit and conjugate with the
            # current Pauli to determine the correct measurement qubits and coefficient.
            measure_pauli = [p for p in mermin_pauli]
            parity = 1
            for op in measurement_circuit.all_operations():
                if isinstance(op.gate, cirq.ops.MeasurementGate):
                    break

                substr = [measure_pauli[cast(cirq.LineQubit, qubit).x] for qubit in op.qubits]
                conjugated_substr = conjugation_rules[op.gate]["".join(substr)]

                if conjugated_substr[0] == "-":
                    parity = -1 * parity
                    conjugated_substr = conjugated_substr[1:]

                for qubit, pauli in zip(op.qubits, conjugated_substr):
                    measure_pauli[cast(cirq.LineQubit, qubit).x] = pauli

            measurement_qubits = [i for i, pauli in enumerate(measure_pauli) if pauli == "Z"]
            measurement_coef = parity

            numerator = 0.0
            for bitstr, count in counts.items():
                parity = 1
                for qb in measurement_qubits:
                    if bitstr[qb] == "1":  # Qubit order is big endian
                        parity = -1 * parity

                numerator += mermin_coef * measurement_coef * parity * count

            expect_val += numerator / sum(list(counts.values()))

        return (expect_val + 2 ** (self.num_qubits - 1)) / 2**self.num_qubits

    def _mermin_operator(self, num_qubits: int) -> List[Tuple[float, str]]:
        """
        Generate the Mermin operator (https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.65.1838),
        or M_n (Eq. 2.8) in https://arxiv.org/pdf/2005.11271.pdf
        """
        x = sympy.symbols("x_1:{}".format(num_qubits + 1))
        y = sympy.symbols("y_1:{}".format(num_qubits + 1))

        term1 = 1
        term2 = 1
        for j in range(num_qubits):
            term1 = term1 * (x[j] + sympy.I * y[j])
            term2 = term2 * (x[j] - sympy.I * y[j])
        term1 = sympy.expand(term1)
        term2 = sympy.expand(term2)

        M_n = (1 / (2 * sympy.I)) * (term1 - term2)
        M_n = sympy.simplify(M_n)

        variables = M_n.as_terms()[1]
        mermin_op = []
        for term in M_n.as_terms()[0]:
            coef = term[1][0][0]
            pauli = [""] * num_qubits
            for i, v in enumerate(term[1][1]):
                if v == 1:
                    char, idx = str(variables[i]).split("_")
                    pauli[int(idx) - 1] = char.upper()

            mermin_op.append((coef, "".join(pauli)))

        return mermin_op

    def _get_measurement_circuit(self) -> stabilizers.MeasurementCircuit:
        """Return a MeasurementCircuit for simultaneous measurement of N operators.

        Each column of self.stabilizer represents a Pauli string that we seek to measure.
        Thus, self.stabilizer should have dimensions of 2 * N rows by N columns. The first N rows
        indicate the presence of a Z in each index of the Pauli String. The last N rows
        indicate X's.

        For instance, simultaneous measurement of YYI, XXY, IYZ would be represented by
        [[1, 0, 0],  ========
         [1, 0, 1],  Z matrix
         [0, 1, 1],  ========
         [1, 1, 0],  ========
         [1, 1, 1],  X matrix
         [0, 1, 0]   ========

        As annotated above, the submatrix of the first (last) N rows is referred to as
        the Z (X) matrix.

        All operators must commute and be independent (i.e. can't express any column as a base-2
        product of the other columns) for this code to work.
        """
        # Validate that the stabilizer matrix is valid
        assert self.stabilizer.shape == (
            2 * self.num_qubits,
            self.num_qubits,
        ), f"{self.num_qubits} qubits, but matrix shape: {self.stabilizer.shape}"

        # i, j will always denote row, column index
        for i in range(2 * self.num_qubits):
            for j in range(self.num_qubits):
                value = self.stabilizer[i, j]
                assert value in [0, 1], f"[{i}, {j}] index is {value}"

        measurement_circuit = stabilizers.MeasurementCircuit(
            cirq.Circuit(), self.stabilizer, self.num_qubits, self.qubits
        )

        stabilizers.prepare_X_matrix(measurement_circuit)
        stabilizers.row_reduce_X_matrix(measurement_circuit)
        stabilizers.patch_Z_matrix(measurement_circuit)
        stabilizers.change_X_to_Z_basis(measurement_circuit)
        # terminate with measurements
        measurement_circuit.get_circuit().append(cirq.measure(*self.qubits))

        return measurement_circuit
