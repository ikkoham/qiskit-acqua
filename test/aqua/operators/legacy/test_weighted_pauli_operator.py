# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test WeightedPauliOperator """

import itertools
import os
import unittest
from test.aqua import QiskitAquaTestCase

import numpy as np
from ddt import ddt, idata, unpack
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Pauli, state_fidelity

from qiskit import BasicAer, QuantumCircuit, QuantumRegister
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.operators import I, WeightedPauliOperator, X, Y, Z
from qiskit.aqua.operators.legacy import op_converter


@ddt
class TestWeightedPauliOperator(QiskitAquaTestCase):
    """WeightedPauliOperator tests."""

    def setUp(self):
        super().setUp()
        seed = 3
        aqua_globals.random_seed = seed

        self.num_qubits = 3
        paulis = [
                Pauli(label="IIZ"),
                Pauli(label="IIY"),
                Pauli(label="IIX"),
                ]
        weights = aqua_globals.random.random(len(paulis))
        self.qubit_op = WeightedPauliOperator.from_list(paulis, weights)
        self.var_form = EfficientSU2(self.qubit_op.num_qubits, reps=1)

    def test_evaluate_with_aer_mode(self):
        """ evaluate with aer mode test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(
                "Aer doesn't appear to be installed. Error: '{}'".format(str(ex))
            )
            return

        seed = 3
        statevector_simulator = Aer.get_backend("statevector_simulator")
        quantum_instance_statevector = QuantumInstance(
            statevector_simulator, shots=1, seed_simulator=seed, seed_transpiler=seed
        )

        wave_function = self.var_form.assign_parameters(
            np.array(aqua_globals.random.standard_normal(self.var_form.num_parameters))
        )

        circuits = self.qubit_op.construct_evaluation_circuit(
            wave_function=wave_function, statevector_mode=True
        )
        for c in circuits:
            print(c.qasm())
        reference = self.qubit_op.evaluate_with_result(
            result=quantum_instance_statevector.execute(circuits), statevector_mode=True
        )

        circuits = self.qubit_op.construct_evaluation_circuit(
            wave_function=wave_function,
            statevector_mode=True,
            use_simulator_snapshot_mode=True,
        )
        actual_value = self.qubit_op.evaluate_with_result(
            result=quantum_instance_statevector.execute(circuits),
            statevector_mode=True,
            use_simulator_snapshot_mode=True,
        )
        self.assertAlmostEqual(reference[0], actual_value[0], places=10)


if __name__ == "__main__":
    unittest.main()
