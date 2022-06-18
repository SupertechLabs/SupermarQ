import cirq
import qiskit

import supermarq as sm


qiskit_circuit = qiskit.QuantumCircuit(2, 1)
qiskit_circuit.swap(0, 1)
qiskit_circuit.measure(0, 0)
qiskit_circuit.reset(0)
qiskit_circuit.measure_all()
cirq_circuit = cirq.Circuit(
    cirq.SWAP(*cirq.LineQubit.range(2)),
    cirq.measure(cirq.LineQubit(0)),
    cirq.reset(cirq.LineQubit(0)),
    cirq.measure(*cirq.LineQubit.range(2)),
)


def test_compute_communication() -> None:
    qiskit_feature = sm.features.compute_communication(qiskit_circuit)
    cirq_feature = sm.features.compute_communication(cirq_circuit)
    assert qiskit_feature >= 0 and qiskit_feature <= 1
    assert cirq_feature >= 0 and cirq_feature <= 1


def test_compute_liveness() -> None:
    qiskit_feature = sm.features.compute_liveness(qiskit_circuit)
    cirq_feature = sm.features.compute_liveness(cirq_circuit)
    assert qiskit_feature >= 0 and qiskit_feature <= 1
    assert cirq_feature >= 0 and cirq_feature <= 1


def test_compute_parallelism() -> None:
    qiskit_feature = sm.features.compute_parallelism(qiskit_circuit)
    cirq_feature = sm.features.compute_parallelism(cirq_circuit)
    assert qiskit_feature >= 0 and qiskit_feature <= 1
    assert cirq_feature >= 0 and cirq_feature <= 1


def test_compute_measurement() -> None:
    qiskit_feature = sm.features.compute_measurement(qiskit_circuit)
    cirq_feature = sm.features.compute_measurement(cirq_circuit)
    assert qiskit_feature >= 0 and qiskit_feature <= 1
    assert cirq_feature >= 0 and cirq_feature <= 1


def test_compute_entanglement() -> None:
    qiskit_feature = sm.features.compute_entanglement(qiskit_circuit)
    cirq_feature = sm.features.compute_entanglement(cirq_circuit)
    assert qiskit_feature >= 0 and qiskit_feature <= 1
    assert cirq_feature >= 0 and cirq_feature <= 1


def test_compute_depth() -> None:
    qiskit_feature = sm.features.compute_depth(qiskit_circuit)
    cirq_feature = sm.features.compute_depth(cirq_circuit)
    assert qiskit_feature >= 0 and qiskit_feature <= 1
    assert cirq_feature >= 0 and cirq_feature <= 1

    qubits = cirq.LineQubit.range(4)
    test_circuit = cirq.Circuit(
        cirq.CX(qubits[0], qubits[1]),
        cirq.CZ(qubits[2], qubits[3]),
        cirq.CX(qubits[1], qubits[2]),
        cirq.CX(qubits[2], qubits[3]),
    )
    test_feature = sm.features.compute_depth(test_circuit)
    assert test_feature >= 0 and test_feature <= 1

    assert sm.features.compute_depth(cirq.Circuit()) == 0
