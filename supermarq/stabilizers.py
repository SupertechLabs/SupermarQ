import copy
from typing import List, Tuple

import cirq
import numpy as np


class MeasurementCircuit:
    def __init__(
        self,
        circuit: cirq.Circuit,
        stabilizer_matrix: np.ndarray,
        num_qubits: int,
        qubits: List[cirq.LineQubit],
    ) -> None:
        self.circuit = circuit
        self.stabilizer_matrix = stabilizer_matrix
        self.num_qubits = num_qubits
        self.qubits = qubits

    def get_circuit(self) -> cirq.Circuit:
        return self.circuit

    def get_stabilizer(self) -> np.ndarray:
        return self.stabilizer_matrix

    def set_circuit(self, circuit: cirq.Circuit) -> None:
        self.circuit = circuit

    def set_stabilizer(self, stabilizer_matrix: np.ndarray) -> None:
        self.stabilizer_matrix = stabilizer_matrix


def construct_stabilizer(N: int, clique: List) -> Tuple[np.ndarray, List]:
    """Construct the independent Z+X stabilizer matrix for the given clique.

    All of the term in the input clique can be measured simultaneously. To construct
    the circuit which will perform this measurement we need to find an independent
    basis for these terms. We'll do this by constructing the Z+X stabilizer matrix
    using all of the terms then select an independent basis using binary Gaussian
    elimination.

    Args:
      N:
        The length of thr clique list.
      clique:
        A list for the clique to turn into a matrix.

    Returns:
      A tuple containing the matrix and pauli basis list for the matrix.
    """
    # Construct the stabilizer matrix as rows
    stabilizer_rows = []
    for _, pauli in clique:
        cur_row = np.zeros(2 * N)
        for i in range(N):
            # Pauli Z's never appear in the Mermin operator
            if pauli[i] == "X":
                cur_row[i + N] = 1
            elif pauli[i] == "Y":
                cur_row[i] = 1
                cur_row[i + N] = 1
        stabilizer_rows.append(cur_row)

    # Transpose and find the independent columns
    dependent_stabilizer = np.array(stabilizer_rows, dtype=np.uint8).T

    # Put the matrix in row Echelon form (mod 2)
    echelon_matrix = binary_gaussian_elimination(copy.copy(dependent_stabilizer))

    # Keep only the independent columns
    stabilizer_matrix = []
    pauli_basis = []
    start_idx = 0
    for i, row in enumerate(echelon_matrix.T):
        if 1 in row[start_idx:]:
            stabilizer_matrix.append(dependent_stabilizer.T[i])
            pauli_basis.append(clique[i][1])
            last_1_idx = 0
            for j, val in enumerate(row):
                if val == 1:
                    last_1_idx = j
            start_idx = last_1_idx + 1

    return np.array(stabilizer_matrix).T, pauli_basis


def binary_gaussian_elimination(M: np.ndarray) -> np.ndarray:
    """Use binary Gaussian elimination to put the input matrix in row Echelon form.

    The input matrix should be an NxM binary matrix.

    Args:
      M:
        The input matrix.

    Returns:
      A matrix in row Echelon form.
    """
    num_rows, num_cols = M.shape
    for i in range(num_cols):
        if i >= num_rows:
            break
        # find the last non-zero element of column i, starting in row i
        max_i = i
        for k in range(i, num_rows):
            if M[k, i] == 1:
                max_i = k

        # Swap rows i and max_i
        M[[i, max_i], :] = M[[max_i, i], :]

        for u in range(i + 1, num_rows):
            # Add M[u, i] * row i to row u
            M[u, :] = (M[u, i] * M[i, :] + M[u, :]) % 2

    # Matrix M is now in row echelon form
    return M


def prepare_X_matrix(measurement_circuit: MeasurementCircuit) -> None:
    """Apply H's to a subset of qubits to ensure that the X matrix has full rank."""
    # TODO: right now, this is naively trying all possibilities. Really, should apply
    # the polytime technique described in Aaronson https://arxiv.org/pdf/quant-ph/0406196.pdf
    N = measurement_circuit.num_qubits
    for bitstring in range(2**N):
        measurement_circuit_copy = copy.deepcopy(measurement_circuit)
        for i, bit in enumerate("{0:b}".format(bitstring).zfill(N)):
            if bit == "1":
                apply_H(measurement_circuit_copy, i)
        if (
            np.linalg.matrix_rank(measurement_circuit_copy.get_stabilizer()[N:]) == N
        ):  # done if full rank
            measurement_circuit.set_circuit(measurement_circuit_copy.get_circuit())
            measurement_circuit.set_stabilizer(measurement_circuit_copy.get_stabilizer())
            return


def row_reduce_X_matrix(measurement_circuit: MeasurementCircuit) -> None:
    """Use Gaussian elimination to reduce the Z matrix to the Identity matrix."""
    transform_X_matrix_to_row_echelon_form(measurement_circuit)
    transform_X_matrix_to_reduced_row_echelon_form(measurement_circuit)


def transform_X_matrix_to_row_echelon_form(measurement_circuit: MeasurementCircuit) -> None:
    N = measurement_circuit.num_qubits
    for j in range(N):
        if measurement_circuit.get_stabilizer()[j + N, j] == 0:
            i = j + 1
            while measurement_circuit.get_stabilizer()[i + N, j] == 0:
                i += 1
            apply_SWAP(measurement_circuit, i, j)

        for i in range(N + j + 1, 2 * N):
            if measurement_circuit.get_stabilizer()[i, j] == 1:
                apply_CNOT(measurement_circuit, j, i - N)


def transform_X_matrix_to_reduced_row_echelon_form(measurement_circuit: MeasurementCircuit) -> None:
    N = measurement_circuit.num_qubits
    for j in range(N - 1, 0, -1):
        for i in range(N, N + j):
            if measurement_circuit.get_stabilizer()[i, j] == 1:
                apply_CNOT(measurement_circuit, j, i - N)


def patch_Z_matrix(measurement_circuit: MeasurementCircuit) -> None:
    stabilizer_matrix, N = measurement_circuit.get_stabilizer(), measurement_circuit.num_qubits
    assert np.allclose(
        stabilizer_matrix[:N], stabilizer_matrix[:N].T
    ), f"Z-matrix,\n{stabilizer_matrix} is not symmetric"

    for i in range(N):
        for j in range(0, i):
            if stabilizer_matrix[i, j] == 1:
                apply_CZ(measurement_circuit, i, j)

        j = i
        if stabilizer_matrix[i, j] == 1:
            apply_S(measurement_circuit, i)


def change_X_to_Z_basis(measurement_circuit: MeasurementCircuit) -> None:
    """Change each qubit from X basis to Z basis via H"""
    N = measurement_circuit.num_qubits
    for j in range(N):
        apply_H(measurement_circuit, j)


def apply_H(measurement_circuit: MeasurementCircuit, i: int) -> None:
    N = measurement_circuit.num_qubits
    qubits = measurement_circuit.qubits
    measurement_circuit.get_stabilizer()[[i, i + N]] = measurement_circuit.get_stabilizer()[
        [i + N, i]
    ]
    measurement_circuit.get_circuit().append(cirq.H(qubits[i]))


def apply_S(measurement_circuit: MeasurementCircuit, i: int) -> None:
    qubits = measurement_circuit.qubits
    measurement_circuit.get_stabilizer()[i, i] = 0
    measurement_circuit.get_circuit().append(cirq.S(qubits[i]))


def apply_CZ(measurement_circuit: MeasurementCircuit, i: int, j: int) -> None:
    qubits = measurement_circuit.qubits
    measurement_circuit.get_stabilizer()[i, j] = 0
    measurement_circuit.get_stabilizer()[j, i] = 0
    measurement_circuit.get_circuit().append(cirq.CZ(qubits[i], qubits[j]))


def apply_CNOT(
    measurement_circuit: MeasurementCircuit, control_index: int, target_index: int
) -> None:
    N = measurement_circuit.num_qubits
    qubits = measurement_circuit.qubits
    measurement_circuit.get_stabilizer()[control_index] = (
        measurement_circuit.get_stabilizer()[control_index]
        + measurement_circuit.get_stabilizer()[target_index]
    ) % 2
    measurement_circuit.get_stabilizer()[target_index + N] = (
        measurement_circuit.get_stabilizer()[control_index + N]
        + measurement_circuit.get_stabilizer()[target_index + N]
    ) % 2
    measurement_circuit.get_circuit().append(cirq.CNOT(qubits[control_index], qubits[target_index]))


def apply_SWAP(measurement_circuit: MeasurementCircuit, i: int, j: int) -> None:
    N = measurement_circuit.num_qubits
    qubits = measurement_circuit.qubits
    measurement_circuit.get_stabilizer()[[i, j]] = measurement_circuit.get_stabilizer()[[j, i]]
    measurement_circuit.get_stabilizer()[[i + N, j + N]] = measurement_circuit.get_stabilizer()[
        [j + N, i + N]
    ]
    measurement_circuit.get_circuit().append(cirq.SWAP(qubits[i], qubits[j]))
