"""Core logic for the 3x3 Sudoku Grover solver (ported from the lab notebook).

Everything here is plain Python/Qiskit with no Streamlit dependency, so it can
be tested on its own.  All backends are credential-free so the app runs on
Streamlit Community Cloud.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from functools import lru_cache
from dataclasses import dataclass, field

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXVChain
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

VALUES = (0, 1, 2)
PERMS = list(itertools.permutations(VALUES))

ALL_CELLS = [(r, c) for r in range(3) for c in range(3)]

# All 12 valid 3x3 Latin squares (rows and columns each contain 0,1,2 once).
LATIN_SQUARES = [
    rows for rows in itertools.product(PERMS, repeat=3)
    if all({rows[r][c] for r in range(3)} == set(VALUES) for c in range(3))
]

# Difficulty = number of revealed clues.  Clue *positions* are random anywhere on
# the grid.  Easy puzzles are additionally limited to 4 search qubits, so a layout
# with all three clues in one row (6 qubits) can never be "easy".
DIFFICULTY = {
    "easy": {"clues": 3, "max_qubits": 4},
    "medium": {"clues": 2, "max_qubits": 6},
}


def make_puzzle(difficulty: str, seed: int | None = None) -> dict:
    """Random puzzle: clues at random cells, taken from a random solved grid.

    Because the clues come from a real solution, the puzzle always has at least
    one valid solution.
    """
    rng = random.Random(seed)
    cfg = DIFFICULTY[difficulty]
    for _ in range(1000):
        solution = rng.choice(LATIN_SQUARES)
        cells = rng.sample(ALL_CELLS, k=cfg["clues"])
        fixed = {(r, c): solution[r][c] for r, c in cells}
        if _qubit_count(fixed) <= cfg["max_qubits"]:
            return fixed
    raise RuntimeError("Could not generate a puzzle within the qubit budget.")


def _row_choices(fixed: dict) -> list:
    return [
        [p for p in PERMS if all(p[c] == v for (rr, c), v in fixed.items() if rr == r)]
        for r in range(3)
    ]


def _width(n_choices: int) -> int:
    return 0 if n_choices <= 1 else math.ceil(math.log2(n_choices))


def _qubit_count(fixed: dict) -> int:
    return sum(_width(len(ch)) for ch in _row_choices(fixed))


# --------------------------------------------------------------------------
# Problem encoding
# --------------------------------------------------------------------------
@dataclass
class Problem:
    fixed: dict
    row_choices: list
    row_widths: list
    offsets: list
    num_qubits: int
    marked: list = field(default_factory=list)

    @property
    def search_space(self) -> int:
        return 2 ** self.num_qubits

    def decode(self, bits):
        """bits: tuple in q0,q1,... order -> list of rows, or None if invalid code."""
        rows = []
        for off, w, choices in zip(self.offsets, self.row_widths, self.row_choices):
            idx = sum(b << i for i, b in enumerate(bits[off:off + w]))
            if idx >= len(choices):
                return None
            rows.append(choices[idx])
        return rows

    def is_valid(self, rows) -> bool:
        if rows is None:
            return False
        return all({rows[r][c] for r in range(3)} == set(VALUES) for c in range(3))

    def key_to_bits(self, key: str):
        """Qiskit count key (c_{n-1}..c_0) -> (q0, q1, ...) tuple."""
        return tuple(int(b) for b in key.strip().replace(" ", "")[::-1][: self.num_qubits])


def build_problem(fixed: dict) -> Problem:
    row_choices = []
    for r in range(3):
        ch = [p for p in PERMS if all(p[c] == v for (rr, c), v in fixed.items() if rr == r)]
        if not ch:
            raise ValueError(f"No legal permutations remain for row {r}.")
        row_choices.append(ch)
    widths = [_width(len(c)) for c in row_choices]
    offsets = [0]
    for w in widths[:-1]:
        offsets.append(offsets[-1] + w)
    prob = Problem(fixed, row_choices, widths, offsets, sum(widths))
    for bits in itertools.product([0, 1], repeat=prob.num_qubits):
        if prob.is_valid(prob.decode(bits)):
            prob.marked.append(bits)
    return prob


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------
def _mcx(qc, controls, target, ancillas):
    k = len(controls)
    if k == 0:
        qc.x(target)
    elif k == 1:
        qc.cx(controls[0], target)
    elif k == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        need = k - 2
        if len(ancillas) < need:
            raise ValueError(f"V-chain MCX needs {need} clean ancillas.")
        qc.append(MCXVChain(k, dirty_ancillas=False),
                  list(controls) + [target] + list(ancillas[:need]))


def _phase_on_bitstring(qc, qubits, bits, ancillas):
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        _mcx(qc, qubits[:-1], qubits[-1], ancillas)
        qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)


def grover_success_probability(num_states: int, num_marked: int, iterations: int) -> float:
    """Ideal (noiseless) probability of measuring a valid solution."""
    theta = math.asin(math.sqrt(num_marked / num_states))
    return math.sin((2 * iterations + 1) * theta) ** 2


def grover_iterations(num_states: int, num_marked: int) -> int:
    """Iteration count (>= 1) that maximises the ideal success probability.

    The textbook pi/4*sqrt(N/M) overshoots when M/N is large (e.g. N=8, M=2), so
    the best count is found directly.
    """
    best, best_p = 1, -1.0
    for k in range(1, 40):
        p = grover_success_probability(num_states, num_marked, k)
        if p > best_p + 1e-12:
            best, best_p = k, p
        elif p < best_p - 1e-12:
            break  # past the first peak
    return best


def build_grover(prob: Problem, iterations: int | None = None) -> QuantumCircuit:
    n = prob.num_qubits
    n_anc = max(0, n - 3)
    iters = iterations or grover_iterations(prob.search_space, len(prob.marked))
    data = list(range(n))
    anc = list(range(n, n + n_anc))
    qc = QuantumCircuit(n + n_anc, n, name="compact_grover")
    qc.h(data)
    for _ in range(iters):
        for bits in prob.marked:                       # oracle
            _phase_on_bitstring(qc, data, bits, anc)
        qc.h(data); qc.x(data)                         # diffuser
        _phase_on_bitstring(qc, data, tuple([1] * n), anc)
        qc.x(data); qc.h(data)
    qc.measure(data, range(n))
    return qc


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def two_qubit_depth(circuit) -> int:
    dag = circuit_to_dag(circuit)
    return sum(
        1 for layer in dag.layers()
        if any(len(node.qargs) >= 2 for node in layer["graph"].op_nodes())
    )


def circuit_metrics(circuit) -> dict:
    c = circuit.remove_final_measurements(inplace=False)
    ops = dict(c.count_ops())
    active = {c.find_bit(q).index for inst in c.data for q in inst.qubits}
    twoq = sum(1 for inst in c.data if len(inst.qubits) == 2)
    return {
        "register width": circuit.num_qubits,
        "active qubits": len(active),
        "total depth": c.depth(),
        "2Q depth": two_qubit_depth(c),
        "2Q gates": twoq,
        "total gates": sum(ops.values()),
        "ops": ops,
    }


# --------------------------------------------------------------------------
# Backends (all credential-free)
# --------------------------------------------------------------------------
IONQ_BASIS = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "swap", "measure"]


@lru_cache(maxsize=1)
def _ibm_fake_sim(device: str):
    """Noisy Aer simulator for one IBM fake device.

    Each 120-156 qubit noise model is slow to build and costs ~300-400 MB, so only the most
    recently used device is kept (maxsize=1). This bounds memory on small hosts such as
    Streamlit Community Cloud, at the price of a rebuild when switching device.
    """
    if device == "marrakesh":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh as Fake
    elif device == "boston":
        from fake_ibm_extra import FakeBoston as Fake
    elif device == "miami":
        from fake_ibm_extra import FakeMiami as Fake
    else:
        raise KeyError(device)
    return AerSimulator.from_backend(Fake())


@lru_cache(maxsize=1)
def _iqm_backend_and_sim():
    """IQM Garnet (20 qubits, native r/cz) with the noise model IQM ships with its fake backend.

    The noise comes from IQM's error profile (per-qubit T1/T2, gate depolarizing errors, readout
    errors, gate durations). AerSimulator.from_backend would drop it, so the model is passed explicitly.
    """
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    be = IQMFakeGarnet()
    return be, AerSimulator(noise_model=be.noise_model)


# --------------------------------------------------------------------------
# Noisy simulators: Cirq, and Quantinuum (pytket compiler + PECOS emulator).
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cirq_google_noise():
    """Google 'willow_pink' noise properties that ship with cirq-google (median device calibration)."""
    import cirq_google as cg
    from cirq_google.engine import load_device_noise_properties

    props = load_device_noise_properties("willow_pink")
    qubits = sorted(props.t1_ns)
    index = {q: i for i, q in enumerate(qubits)}
    edges = [[index[a], index[b]] for a in qubits for b in qubits if a < b and a.is_adjacent(b)]
    edges += [[b, a] for a, b in edges]
    return qubits, CouplingMap(edges), cg.NoiseModelFromGoogleNoiseProperties(props)


def _run_cirq(circuit, shots, seed):
    """Cirq with the noise model and median calibration of Google's 'willow_pink' device (cirq-google).

    The circuit is routed onto the device's grid, every CX is rewritten into CZ gates
    (CZ), and Cirq's NoiseModelFromGoogleNoiseProperties adds gate Pauli errors,
    T1/Tphi decay and readout errors before a density-matrix simulation.
    """
    import cirq
    from collections import Counter

    qubits, cmap, model = _cirq_google_noise()
    tc = transpile(circuit, coupling_map=cmap, basis_gates=["rz", "sx", "x", "cx", "measure"],
                   optimization_level=3, seed_transpiler=seed)
    ops, meas = [], {}
    for inst in tc.data:
        name = inst.operation.name
        gq = [qubits[tc.find_bit(q).index] for q in inst.qubits]
        if name == "rz":
            ops.append(cirq.rz(float(inst.operation.params[0])).on(gq[0]))
        elif name == "sx":
            ops.append((cirq.X ** 0.5).on(gq[0]))
        elif name == "x":
            ops.append(cirq.X.on(gq[0]))
        elif name == "cx":
            ops.append(cirq.CNOT.on(*gq))
        elif name == "measure":
            meas[tc.find_bit(inst.clbits[0]).index] = gq[0]
        elif name != "barrier":
            raise ValueError(f"Unexpected gate {name}")
    ccirc = cirq.optimize_for_target_gateset(cirq.Circuit(ops), gateset=cirq.CZTargetGateset())
    cols = sorted(meas)
    ccirc.append(cirq.measure(*[meas[c] for c in cols], key="m"))
    res = cirq.DensityMatrixSimulator(noise=model, seed=seed).run(ccirc, repetitions=shots)
    counts = Counter("".join(str(int(b)) for b in reversed(row)) for row in res.measurements["m"])
    return tc, dict(counts)


# Typical values from the Quantinuum H2 Product Data Sheet v2.00 (15 Oct 2024):
# 1q infidelity 3e-5, 2q infidelity 1.5e-3, SPAM error 1.5e-3 (split evenly between
# preparation and measurement). Memory and cross-talk errors are not modelled.
QTM_ERR = {"p1": 3e-5, "p2": 1.5e-3, "p_meas": 7.5e-4, "p_init": 7.5e-4}


def _run_quantinuum(circuit, shots, seed):
    """pytket-quantinuum compiles to H2 native gates; PECOS emulates it with H-series noise."""
    from pecos.error_models.depolarizing_error_model import DepolarizingErrorModel
    from pytket.backends.backendresult import BackendResult
    from pytket.extensions.quantinuum import QuantinuumBackend
    from pytket.qasm import circuit_from_qasm_str
    from pytket_pecos import Emulator
    from qiskit import qasm2

    tc = transpile(circuit, basis_gates=["rz", "sx", "x", "cx", "measure"],
                   optimization_level=3, seed_transpiler=seed)
    be = QuantinuumBackend("H2-1LE")  # local emulator device, no account or network needed
    native = be.get_compiled_circuit(circuit_from_qasm_str(qasm2.dumps(tc)), 2)
    em = Emulator(native, error_model=DepolarizingErrorModel(error_params=dict(QTM_ERR)),
                  qsim="state-vector", seed=seed)
    res = BackendResult(c_bits=native.bits, shots=em.run(n_shots=shots))
    from collections import Counter
    counts = Counter()
    for bits, v in res.get_counts().items():
        counts["".join(str(int(x)) for x in reversed(bits))] += v
    return tc, dict(counts)


BACKENDS = {
    "Cirq noisy": {
        "vendor": "Google Cirq (Willow calibration)",
        "noise": "Noisy",
        "native": "CZ, PhasedXZ (105-qubit grid)",
        "desc": "Cirq NoiseModelFromGoogleNoiseProperties with the median calibration of Google's Willow device shipped in cirq-google (gate Pauli errors, T1/Tphi, readout), after routing onto the device grid and rewriting CX into CZ.",
    },
    "Quantinuum noisy": {
        "vendor": "Quantinuum (H2 emulator)",
        "noise": "Noisy",
        "native": "ZZPhase, PhasedX, Rz (all-to-all)",
        "desc": "Compiled with Quantinuum's pytket compiler for H2 native gates and run on the PECOS emulator with the typical error figures from the H2 product data sheet (1q 3e-5, 2q 1.5e-3, SPAM 1.5e-3). Local emulator, no account needed.",
    },
    "Aer ideal": {
        "vendor": "Qiskit Aer",
        "noise": "Noiseless",
        "native": "rz, sx, x, cx",
        "desc": "Noiseless local statevector simulation, transpiled to {rz, sx, x, cx} at optimization level 3.",
    },
    "IonQ-style ideal": {
        "vendor": "IonQ (emulated)",
        "noise": "Noiseless",
        "native": "h, t, tdg, rx/ry/rz, cx, ...",
        "desc": "Transpiled to IonQ's QIS gate set (optimization level 0, as in the notebook) and run on Aer. All-to-all connectivity, so no SWAPs.",
    },
    "IQM Garnet noisy": {
        "vendor": "IQM (fake backend)",
        "noise": "Noisy",
        "native": "r, cz",
        "desc": "IQMFakeGarnet: 20-qubit device with IQM's native {r, cz} gates and the noise data IQM ships with it (T1/T2, gate and readout errors).",
    },
    "IBM Marrakesh noisy": {
        "vendor": "IBM (fake backend)",
        "noise": "Noisy",
        "native": "sx, rz, x, cz (heavy-hex)",
        "desc": "156-qubit ibm_marrakesh snapshot (calibration noise model) run on Aer, transpiled at optimization level 3.",
    },
    "IBM Boston noisy": {
        "vendor": "IBM (fake backend)",
        "noise": "Noisy",
        "native": "sx, rz, x, cz",
        "desc": "156-qubit ibm_boston snapshot (calibration noise model) run on Aer, transpiled at optimization level 3.",
    },
    "IBM Miami noisy": {
        "vendor": "IBM (fake backend)",
        "noise": "Noisy",
        "native": "sx, rz, x, cz (square lattice)",
        "desc": "120-qubit ibm_miami snapshot (calibration noise model) run on Aer, transpiled at optimization level 3.",
    },
}


def run_backend(name: str, circuit: QuantumCircuit, shots: int, seed: int = 1234):
    """Returns (transpiled_circuit, counts, runtime_seconds)."""
    t0 = time.time()
    if name == "Aer ideal":
        tc = transpile(circuit, basis_gates=["rz", "sx", "x", "cx", "measure"],
                       optimization_level=3, seed_transpiler=seed)
        res = AerSimulator().run(tc, shots=shots, seed_simulator=seed).result()
    elif name == "IonQ-style ideal":
        tc = transpile(circuit, basis_gates=IONQ_BASIS, optimization_level=0, seed_transpiler=seed)
        res = AerSimulator().run(tc, shots=shots, seed_simulator=seed).result()
    elif name == "IQM Garnet noisy":
        be, sim = _iqm_backend_and_sim()
        tc = transpile(circuit, backend=be, optimization_level=3, seed_transpiler=seed)
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    elif name == "IBM Marrakesh noisy":
        sim = _ibm_fake_sim("marrakesh")
        tc = transpile(circuit, backend=sim, optimization_level=3, seed_transpiler=seed)
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    elif name in ("IBM Boston noisy", "IBM Miami noisy"):
        sim = _ibm_fake_sim("boston" if name == "IBM Boston noisy" else "miami")
        tc = transpile(circuit, backend=sim, optimization_level=3, seed_transpiler=seed)
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    elif name == "Cirq noisy":
        tc, counts = _run_cirq(circuit, shots, seed)
        return tc, counts, time.time() - t0
    elif name == "Quantinuum noisy":
        tc, counts = _run_quantinuum(circuit, shots, seed)
        return tc, counts, time.time() - t0
    else:
        raise KeyError(name)
    return tc, dict(res.get_counts()), time.time() - t0


def analyse_counts(prob: Problem, counts: dict) -> dict:
    """Success probability and best decoded grid."""
    shots = sum(counts.values())
    good = 0
    for key, v in counts.items():
        if prob.is_valid(prob.decode(prob.key_to_bits(key))):
            good += v
    best_key = max(counts, key=counts.get)
    best_rows = prob.decode(prob.key_to_bits(best_key))
    return {
        "success": good / shots,
        "best_key": best_key,
        "best_rows": best_rows,
        "best_valid": prob.is_valid(best_rows),
        "shots": shots,
    }
