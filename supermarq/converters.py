import cirq
import qiskit


def cirq_to_qiskit(circuit: cirq.Circuit) -> qiskit.circuit.QuantumCircuit:
    qasm = cirq.circuits.QasmOutput(circuit, tuple(sorted(circuit.all_qubits())))
    return qiskit.circuit.QuantumCircuit().from_qasm_str(str(qasm))
