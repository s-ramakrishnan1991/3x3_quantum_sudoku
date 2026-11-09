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
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

VALUES = (0, 1, 2)
PERMS = list(itertools.permutations(VALUES))

# Clue positions per difficulty (row, col).  Values are drawn at random but
# always distinct, so every generated puzzle is guaranteed solvable.
DIFFICULTY_CELLS = {
    "easy": [(0, 0), (1, 1), (2, 2)],
    "medium": [(0, 0), (1, 1)],
}


def make_puzzle(difficulty: str, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    cells = DIFFICULTY_CELLS[difficulty]
    vals = rng.sample(list(VALUES), k=len(cells))
    return dict(zip(cells, vals))


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
    widths = [max(1, math.ceil(math.log2(len(c)))) for c in row_choices]
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


def grover_iterations(num_states: int, num_marked: int) -> int:
    return max(1, round((math.pi / 4) * math.sqrt(num_states / num_marked)))


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


def _iqm_like_backend(rows=4, cols=5, seed=7):
    """20-qubit square-lattice device with IQM's native {r, cz} gate set and noise."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append([i, i + 1])
            if r + 1 < rows:
                edges.append([i, i + cols])
    return GenericBackendV2(
        num_qubits=rows * cols,
        basis_gates=["r", "cz"],
        coupling_map=CouplingMap(edges),
        noise_info=True,
        seed=seed,
    )


@lru_cache(maxsize=1)
def _ibm_marrakesh_sim():
    """Building the 156-qubit noise model is slow, so do it once per process."""
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    return AerSimulator.from_backend(FakeMarrakesh())


@lru_cache(maxsize=1)
def _iqm_backend_and_sim():
    be = _iqm_like_backend()
    return be, AerSimulator.from_backend(be)


BACKENDS = {
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
    "IQM-style noisy": {
        "vendor": "IQM (emulated)",
        "noise": "Noisy",
        "native": "r, cz",
        "desc": "20-qubit square-lattice device with IQM's {r, cz} native gates and a synthetic noise model.",
    },
    "IBM Marrakesh noisy": {
        "vendor": "IBM (fake backend)",
        "noise": "Noisy",
        "native": "sx, rz, x, cz (heavy-hex)",
        "desc": "156-qubit ibm_marrakesh snapshot (calibration noise model) run on Aer, transpiled at optimization level 3.",
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
    elif name == "IQM-style noisy":
        be, sim = _iqm_backend_and_sim()
        tc = transpile(circuit, backend=be, optimization_level=3, seed_transpiler=seed)
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    elif name == "IBM Marrakesh noisy":
        sim = _ibm_marrakesh_sim()
        tc = transpile(circuit, backend=sim, optimization_level=3, seed_transpiler=seed)
        res = sim.run(tc, shots=shots, seed_simulator=seed).result()
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
