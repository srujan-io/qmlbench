from qiskit import QuantumCircuit


def create_feature_map(x):
    if len(x) != 2:
        raise ValueError("Expected exactly 2 features")

    circuit = QuantumCircuit(2)

    circuit.ry(float(x[0]), 0)
    circuit.ry(float(x[1]), 1)

    return circuit
