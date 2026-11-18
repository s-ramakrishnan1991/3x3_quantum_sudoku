"""IBM FakeBoston and FakeMiami for the pinned qiskit-ibm-runtime.

These snapshots first shipped in qiskit-ibm-runtime 0.47, which needs Qiskit >= 2.2 and so
cannot be installed next to the IQM client (Qiskit < 2.2). Their unmodified data files
(Apache-2.0, copyright IBM 2026) are bundled in ``fake_backends/`` and loaded here with the
FakeBackendV2 base class the pinned runtime provides. See fake_backends/LICENSE.txt.
"""
from __future__ import annotations

import os

from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_backends")


class FakeBoston(FakeBackendV2):
    """Snapshot of IBM's 156-qubit ``ibm_boston``."""

    dirname = os.path.join(_DATA, "boston")
    conf_filename = "conf_boston.json"
    props_filename = "props_boston.json"
    backend_name = "fake_boston"
    _physical_qubits = 332


class FakeMiami(FakeBackendV2):
    """Snapshot of IBM's 120-qubit ``ibm_miami``."""

    dirname = os.path.join(_DATA, "miami")
    conf_filename = "conf_miami.json"
    props_filename = "props_miami.json"
    backend_name = "fake_miami"
    _physical_qubits = 338
