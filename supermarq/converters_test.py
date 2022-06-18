import cirq
import qiskit

import supermarq as sm


def test_cirq_to_qiskit() -> None:
    cirq_circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(0)), cirq.CX(cirq.LineQubit(0), cirq.LineQubit(1))
    )
    qiskit_circuit = qiskit.QuantumCircuit(2)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)
    assert sm.converters.cirq_to_qiskit(cirq_circuit) == qiskit_circuit
