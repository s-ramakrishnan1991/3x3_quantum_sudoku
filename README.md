# Quantum Sudoku: Grover's algorithm on a 3×3 grid

An interactive [Streamlit](https://streamlit.io) app that solves a 3×3 Sudoku with **Grover's search algorithm**,
runs the same circuit on a range of simulated quantum backends, and (optionally) on real quantum hardware
and cloud simulators using your own accounts. It is a web port of the notebook
`03_Grovers_Algorithm_Sudoku.ipynb`.

> **Rules of the puzzle.** Fill the 3×3 grid with the values 0, 1 and 2 so that every row and every column
> contains each value exactly once (a 3×3 Latin square). A few cells are given as clues.

---

## Contents

1. [Features](#features)
2. [How the quantum solver works](#how-the-quantum-solver-works)
3. [Backends and where their noise comes from](#backends-and-where-their-noise-comes-from)
4. [Example results](#example-results)
5. [Real hardware and cloud tab](#real-hardware-and-cloud-tab)
6. [Project structure](#project-structure)
7. [Run locally](#run-locally)
8. [Deploy on Streamlit Community Cloud](#deploy-on-streamlit-community-cloud)
9. [Dependencies and version pins](#dependencies-and-version-pins)
10. [Resource requirements](#resource-requirements)
11. [Troubleshooting](#troubleshooting)
12. [Limitations and honest caveats](#limitations-and-honest-caveats)
13. [Security and privacy](#security-and-privacy)
14. [Licenses](#licenses)

---

## Features

| Tab | What it does |
|---|---|
| 🎮 **Solve** | Pick a difficulty, press **New random puzzle**, choose a backend and run Grover. Shows the puzzle, the measurement histogram, the decoded grid, the success probability and circuit metrics. |
| 📊 **Backend comparison** | Runs the same puzzle on all simulated backends and compares success probability, circuit size, two-qubit gate counts and run time in a table and charts. Finished real-hardware jobs are added to the same table. |
| ☁️ **Hardware & cloud** | Optional. Submit the circuit to IBM Quantum, IQM or IonQ (QPU, or IonQ's noisy cloud simulator) with your own credentials. Jobs run asynchronously and are polled automatically. |
| 🔬 **How it works** | Step-by-step explanation of the encoding, oracle and diffuser for the current puzzle. |
| ℹ️ **Notes** | Backend table, noise provenance, and platform notes. |

Other features:

* **Two difficulty levels.** *Easy* gives 3 clues, *medium* gives 2 clues (fewer clues means a larger search space
  and a deeper circuit).
* **Random clue placement.** Clues can appear anywhere on the grid, not only on the diagonal. A session seed
  makes "New random puzzle" reproducible within a session.
* **Light and dark theme.** Use the app menu (⋮ → Settings → Theme).
* **Adjustable shots** (512 to 8192) and an advanced override for the number of Grover iterations.
* **Cached simulations**, so repeated runs of the same puzzle, backend and shot count return instantly.

---

## How the quantum solver works

### Encoding
Every row of a valid grid is a permutation of (0, 1, 2), so there are only 6 possible rows. After the clues
are applied, each row has *n* remaining candidate permutations, and it is encoded in ⌈log₂ n⌉ qubits
(zero qubits if only one candidate remains). The search register therefore holds one index per row, and the
search space has *N* = product of the per-row candidate counts. Easy puzzles use up to 4 search qubits,
medium puzzles up to 6.

### Oracle
Rows are valid by construction, so the oracle only has to check the **column constraints**: each column must
contain 0, 1 and 2 exactly once. The check is a classical, reversible comparison implemented with multi-controlled
X gates (V-chain decomposition using clean ancilla qubits), and it phase-flips every marked (valid) state.
The register is `search qubits + max(0, search qubits − 3)` ancillas wide (5 qubits for the easy example,
up to 9 for medium).

### Diffuser and iteration count
After the oracle, the standard Grover diffuser amplifies the marked states. With *M* valid solutions out of *N*
states, the ideal success probability after *k* iterations is

```
P(k) = sin²((2k + 1)·θ),   θ = asin(√(M/N))
```

The textbook count π/4·√(N/M) overshoots when M/N is large, so the app searches directly for the **first peak**
of P(k). You can override the iteration count under *Advanced* in the sidebar.

### Success metric
A run counts as successful when a measured bitstring decodes to a **valid, complete grid** consistent with the
clues. The app reports the fraction of shots that did so, the most frequent outcome, and whether that outcome is valid.

---

## Backends and where their noise comes from

All simulators run locally inside the app and need no account. The table says exactly what each one does, because
"noisy simulator" can mean very different things.

| Backend | Noise | Source of the noise / what actually runs |
|---|---|---|
| **Aer ideal** | none | Qiskit Aer statevector-style simulation of the transpiled circuit. The reference. |
| **IonQ-style ideal** | none | Circuit transpiled to an IonQ-like gate set (`h, t, tdg, rx/ry/rz, cx, …`) and simulated without noise. Not IonQ's own simulator. |
| **IQM Garnet noisy** | yes | `IQMFakeGarnet` from `iqm-client`: 20 qubits, native `r` and `cz` gates, with the noise model IQM ships with the fake backend (per-qubit T1/T2, gate depolarizing errors, readout errors, gate durations). |
| **IBM Marrakesh noisy** | yes | `FakeMarrakesh` from `qiskit-ibm-runtime` (156 qubits, heavy-hex), noise model built from the stored calibration snapshot. |
| **IBM Boston noisy** | yes | `FakeBoston` snapshot (156 qubits). Data bundled in `fake_backends/` (see below). |
| **IBM Miami noisy** | yes | `FakeMiami` snapshot (120 qubits, square lattice). Data bundled in `fake_backends/`. |
| **Cirq noisy** | yes | Cirq's `NoiseModelFromGoogleNoiseProperties` with the **median calibration of Google's Willow device** (`willow_pink`, shipped inside `cirq-google`): gate Pauli errors, T1/Tphi decay and readout errors. The circuit is routed onto Willow's 105-qubit grid, CX gates are rewritten into CZ, and the result is simulated with Cirq's density-matrix simulator. |
| **Quantinuum noisy** | yes | `pytket-quantinuum` compiles the circuit for the **H2** native gate set and the PECOS emulator runs it (all-to-all connectivity) with the **typical figures from the H2 Product Data Sheet v2.00 (15 Oct 2024)**: single-qubit infidelity 3×10⁻⁵, two-qubit infidelity 1.5×10⁻³, SPAM error 1.5×10⁻³ (split evenly between preparation and measurement). Memory and cross-talk errors are not modelled. |

Why some backends are missing: CUDA-Q needs about 1.6 GB of CUDA libraries, QuEra's and Pasqal's simulators
cannot run a transpiled gate-model Grover circuit in this setup (Pasqal's Qiskit provider accepts only a single
analog `HamiltonianGate`, no `rz`/`sx`/`cx`), and AWS Braket was removed at the project owner's request.

---

## Example results

Success probability for one test puzzle per difficulty (seed 3, 1024 shots). Other puzzles differ, and real calibration
data changes over time, so treat these as an illustration of the *ordering*, not as benchmarks.

| Backend | Easy | Medium |
|---|---|---|
| Aer ideal | 0.96 | 0.97 |
| Quantinuum noisy (H2 data sheet) | 0.85 | 0.79 |
| Cirq noisy (Google Willow) | 0.53 | 0.25 |
| IQM Garnet noisy | 0.27 | 0.21 |

Medium puzzles need roughly twice as many two-qubit gates as easy ones, so noise hurts them more. Backends
with all-to-all connectivity and low two-qubit error (Quantinuum) fare best; sparse connectivity (extra SWAPs)
and ~1% two-qubit error (IQM, Google) make the circuit much noisier. For reference, Google's older *Rainbow* and *Weber*
calibrations gave results indistinguishable from random guessing on this circuit.

---

## Real hardware and cloud tab

Everything here is optional and runs on **your** account, so it may consume quota, credits or money.
Tick the acknowledgement checkbox before submitting.

| Provider | You enter | Notes |
|---|---|---|
| **IBM Quantum** | API key, CRN (instance), choice of backend | Click *Connect & list backends* to load the backends your instance can use. Uses `SamplerV2` through the `ibm_quantum_platform` channel. |
| **IQM** | API key, choice of quantum computer (`garnet`, `emerald`, `sirius`) | Defaults to the IQM Resonance URL (`https://resonance.iqm.tech/`). |
| **IonQ** | API key, QPU or **noisy simulator** | The noisy cloud simulator lets you pick a noise model (`forte-1`, `forte-enterprise-1`, `aria-2`, `aria-1`). The QPU needs the device to be enabled for your project; otherwise IonQ returns an HTTP 403. |

How jobs work:

* Submitting returns a job ID immediately. The app polls every 15 seconds and shows the status
  (queued, running, done, failed).
* Finished jobs are decoded, scored and added to the **Backend comparison** tab (labelled *Real hardware* or *Cloud simulator*).
* **Retrieve an earlier job by ID** lets you fetch a job you submitted before (the API key is needed again).
* Hardware circuits are transpiled for the device: IBM and IQM use the backend's own target, IonQ uses its native
  gate set with optimization level 0.

---

## Project structure

```
app.py                Main Streamlit app (tabs, sidebar, caching, charts, Notes text)
grover_engine.py      Puzzle generation, Grover circuit builder, metrics, simulated backends
real_backends.py      IBM / IQM / IonQ submission, status polling and result fetching
hardware_tab.py       UI and session state for the Hardware & cloud tab
fake_ibm_extra.py     FakeBoston / FakeMiami backend classes
fake_backends/        Bundled IBM calibration snapshots for Boston and Miami (Apache-2.0)
ui_helpers.py         CSS, HTML grid renderer, result formatting, outcome chart
requirements.txt      Pinned dependencies (see below)
.streamlit/config.toml  Server and light/dark theme settings
.gitignore
```

Key entry points in `grover_engine.py`: `make_puzzle(difficulty, seed)`, `build_problem(fixed)`, `build_grover(prob)`,
`run_backend(name, circuit, shots)` and `analyse_counts(prob, counts)`. New simulators are added by registering an entry
in the `BACKENDS` dictionary and a branch in `run_backend`.

---

## Run locally

Python 3.12 is recommended (3.11 to 3.13 work). **Use a fresh virtual environment**: the pinned Qiskit 2.1.2 must not
be mixed with other Qiskit versions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Plain `pip` can be slow to resolve this dependency set. [`uv`](https://docs.astral.sh/uv/) is much faster and is what
Streamlit Cloud uses:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
python -m streamlit run app.py
```

Run the app with `python -m streamlit` from the same environment you installed into, so that the packages are found.

---

## Deploy on Streamlit Community Cloud

1. Push the files to a GitHub repository **with `app.py` and `requirements.txt` at the repository root** (not inside an
   extra `app/` sub-folder).
2. At <https://share.streamlit.io> choose **Create app**, select the repo and branch, and set **Main file path** to `app.py`.
3. Under **Advanced settings** choose **Python 3.12** (3.11 to 3.13 work), then deploy.
4. After a push, Streamlit redeploys automatically. If the app still shows an older version, open **Manage app → logs**,
   check which commit was cloned, look for install errors, and use **Reboot app**.

The first build takes several minutes. See [Resource requirements](#resource-requirements) before deploying.

---

## Dependencies and version pins

| Package | Why |
|---|---|
| `qiskit==2.1.2`, `qiskit-aer==0.17.2` | Circuit construction, transpilation, Aer simulation. Pinned because `iqm-client[qiskit]` 36 needs `qiskit<2.2`, and newer Qiskit 2.6.x / Aer 0.18.x **crashed (segfault) the Streamlit server on the second rerun** during testing. |
| `qiskit-ibm-runtime==0.43.1` | IBM backends and fake devices. Newer versions that include FakeBoston/FakeMiami need Qiskit ≥ 2.2, hence the bundled data in `fake_backends/`. |
| `qiskit-ionq==1.1.1` | IonQ QPU and cloud simulator. |
| `iqm-client[qiskit]==36.0.0` | IQM fake backends (Garnet) and the real-hardware provider. |
| `ibm-cloud-sdk-core==3.24.1`, `python-dateutil<3` | Needed so that `ibm-platform-services` (pulled in by `qiskit-ibm-runtime`) builds. |
| `cirq-core`, `cirq-google` | Cirq simulation and the Google Willow noise model. |
| `pytket-quantinuum[pecos]==0.59.3`, `quantum-pecos==0.8.0.dev8` | pytket compiler for H2 and the local PECOS emulator. The `[pecos]` extra depends on a **pre-release** of `quantum-pecos`, so it is pinned explicitly. |
| `pandas`, `matplotlib`, `pylatexenc`, `streamlit>=1.50,<2` | Tables, charts, circuit drawing, UI. |

Do not loosen the Qiskit pins without re-testing, and check the environment with `uv pip check` (or `pip check`) after any change.

---

## Resource requirements

* **Disk:** about 1.9 GB for the full environment. Most of it is the Quantinuum PECOS emulator and its toolchain.
* **Memory:** one simulated backend uses a few hundred MB. Running **all** backends in one session peaked at about
  **1.3 GB** in testing, which can exceed Streamlit Community Cloud's limit of roughly 1 GB. The app limits memory by
  keeping only one IBM noise model loaded at a time (switching IBM device takes about 10 seconds), but if the app restarts
  during **Run comparison**, run fewer backends at a time or deploy on a host with more memory.
* **Time:** a simulated run takes from under a second to roughly ten seconds. Cirq with the Willow noise model is the slowest
  on medium puzzles.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Cloud app still shows old backends after a push | Check that `app.py` and `requirements.txt` are at the repo root, that the app watches the branch you pushed to, then look at **Manage app → logs** for the cloned commit and any install error, and **Reboot app**. |
| Build fails or app restarts when running the comparison | Out of disk or memory on the host (see above). Check the logs for `Killed` or `No space left on device`. |
| `ModuleNotFoundError` locally | The packages were installed into a different environment. Reinstall with `pip install -r requirements.txt` in the active venv and start with `python -m streamlit run app.py`. |
| Streamlit server crashes (segfault) on the second run | A newer Qiskit/Aer than the pinned ones is installed. Recreate the environment from `requirements.txt`. |
| IonQ `403 … qpu.forte-1 not enabled` | Your IonQ project does not have that QPU enabled. Use the noisy simulator or ask IonQ to enable the device. |
| IBM backend list is empty | Check the API key and the CRN (instance), and that your plan has access to backends. |
| Only a light theme appears | Use ⋮ → Settings → Theme. The selector appears because `config.toml` defines both `[theme.light]` and `[theme.dark]`. |

---

## Limitations and honest caveats

* The noisy simulators use **stored or published error figures**, not live calibration, so results are indicative.
  The Google Willow data is a *median* over the chip; the Quantinuum figures are the data sheet's *typical* values, and
  memory and cross-talk errors are not modelled.
* **IonQ-style ideal** is not IonQ's simulator; it is a noiseless run with an IonQ-like gate set. Use the Hardware tab's
  IonQ noisy cloud simulator for IonQ's own noise models.
* Fake-backend noise models describe the devices as of their snapshot date.
* The real-hardware and cloud-simulator paths depend on third-party services and were checked offline (request payloads
  and result handling) but not against live accounts, so provider-side errors (permissions, queues, quotas) can occur.
* Success probability is measured over a finite number of shots, so small differences between backends are within noise.
* The puzzle is small by design: 3×3 grid, values 0 to 2, 2 to 3 clues.

---

## Security and privacy

* API keys and CRNs are entered in the UI, held only in the running session, used for the requested call, and **never
  stored, cached, logged or written to environment variables**.
* Job records contain only the provider, backend, job ID, shot count and circuit metrics, never credentials.
* Error messages are scrubbed of secrets before they are shown.
* Because the app runs on a shared server when deployed, use API keys with the minimum permissions you can, and rotate any
  key you pasted into a screenshot or chat.

---

## Licenses

* The app code is provided for the project owner's use; add your own license file if you plan to publish it.
* The IBM FakeBoston and FakeMiami data in `fake_backends/` are unmodified files from the IBM Qiskit ecosystem under the
  **Apache License 2.0** (see `fake_backends/LICENSE.txt`).
* Qiskit, Qiskit Aer, `qiskit-ibm-runtime`, `qiskit-ionq`, `iqm-client`, Cirq, `cirq-google`, pytket and PECOS are
  distributed under their own open-source licenses.
* Google, IBM, IQM, IonQ and Quantinuum names refer to the respective companies' devices and tools; this project is
  not affiliated with or endorsed by them.

*Source notebook: `03_Grovers_Algorithm_Sudoku.ipynb`.*
