"""Optional real-hardware execution on IBM Quantum, IQM and IonQ.

Design notes
------------
* Credentials are passed in by the caller on every call. Nothing here stores,
  caches or logs them, and error messages are scrubbed of the secrets.
* Real QPU jobs queue for minutes to hours, so execution is split into
  ``submit`` (returns a job id immediately) and ``fetch`` (poll status / get
  counts). The UI never blocks waiting on a queue.
* Each provider SDK is imported lazily so the simulator-only parts of the app
  still work if an SDK is missing.
"""
from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile

import grover_engine as ge

PROVIDERS = ["IBM Quantum", "IQM", "IonQ"]

IQM_DEFAULT_URL = "https://resonance.iqm.tech/"
# Common IQM Resonance system aliases; the UI also accepts a custom alias.
IQM_COMPUTERS = ["garnet", "emerald", "sirius"]
IONQ_BACKEND = "ionq_qpu"

FINAL_STATES = {"done", "error", "cancelled"}


class HardwareError(Exception):
    """A user-presentable error (credentials scrubbed)."""


@dataclass
class Submission:
    provider: str
    backend: str
    job_id: str
    shots: int
    metrics: dict  # transpiled-circuit metrics (see grover_engine.circuit_metrics)


# ---------------------------------------------------------------- utilities
def _scrub(text: str, *secrets: str | None) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def _wrap(exc: Exception, *secrets: str | None) -> HardwareError:
    msg = _scrub(f"{type(exc).__name__}: {exc}", *secrets)
    return HardwareError(msg[:600])


def normalize_status(status) -> str:
    """Map provider-specific job states onto queued / running / done / error / cancelled."""
    name = str(getattr(status, "name", status)).upper().replace("JOBSTATUS.", "")
    if name in {"DONE", "COMPLETED", "SUCCEEDED"}:
        return "done"
    if name in {"ERROR", "FAILED"}:
        return "error"
    if name in {"CANCELLED", "CANCELED"}:
        return "cancelled"
    if name in {"RUNNING", "EXECUTING"}:
        return "running"
    return "queued"  # INITIALIZING, VALIDATING, QUEUED, SUBMITTED, ...


# ---------------------------------------------------------------- IBM
def _ibm_service(token: str, crn: str):
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Explicit token + instance (CRN) so no saved account on disk is ever used.
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)


def ibm_list_backends(token: str, crn: str) -> list[dict]:
    """Operational, non-simulator IBM QPUs available to this account/instance."""
    try:
        service = _ibm_service(token, crn)
        out = []
        for be in service.backends(simulator=False, operational=True):
            try:
                pending = be.status().pending_jobs
            except Exception:
                pending = None
            out.append({"name": be.name, "qubits": be.num_qubits, "pending_jobs": pending})
        out.sort(key=lambda d: (d["pending_jobs"] is None, d["pending_jobs"] or 0))
        return out
    except Exception as exc:
        raise _wrap(exc, token, crn) from None


def _ibm_submit(circuit, shots, token, crn, backend_name) -> Submission:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    service = _ibm_service(token, crn)
    backend = service.backend(backend_name)
    isa = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                       seed_transpiler=1234).run(circuit)
    job = SamplerV2(mode=backend).run([isa], shots=shots)
    return Submission("IBM Quantum", backend_name, job.job_id(), shots, ge.circuit_metrics(isa))


def _ibm_fetch(job_id, token, crn, creg):
    service = _ibm_service(token, crn)
    job = service.job(job_id)
    status = normalize_status(job.status())
    if status == "done":
        pub = job.result()[0]
        return status, dict(getattr(pub.data, creg).get_counts()), None
    err = None
    if status == "error":
        try:
            err = str(job.error_message())
        except Exception:
            err = "The job failed on the IBM side."
    return status, None, err


# ---------------------------------------------------------------- IQM
def _iqm_backend(token, url, computer):
    from iqm.qiskit_iqm import IQMProvider

    kwargs = {"quantum_computer": computer} if computer else {}
    return IQMProvider(url or IQM_DEFAULT_URL, token=token, **kwargs).get_backend()


def _iqm_submit(circuit, shots, token, url, computer) -> Submission:
    backend = _iqm_backend(token, url, computer)
    tc = transpile(circuit, backend, optimization_level=3, seed_transpiler=1234)
    job = backend.run(tc, shots=shots)
    return Submission("IQM", computer or "default", job.job_id(), shots, ge.circuit_metrics(tc))


def _iqm_fetch(job_id, token, url, computer):
    job = _iqm_backend(token, url, computer).retrieve_job(job_id)
    status = normalize_status(job.status())
    if status == "done":
        return status, dict(job.result().get_counts()), None
    err = None
    if status == "error":
        try:
            err = str(job.error_message())
        except Exception:
            err = "The job failed on the IQM side."
    return status, None, err


# ---------------------------------------------------------------- IonQ
def _ionq_backend(token, name=None):
    import warnings

    from qiskit_ionq import IonQProvider

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SDK warns about transpile level / qubit lookups
        return IonQProvider(token).get_backend(name or IONQ_BACKEND)


def _ionq_submit(circuit, shots, token, name) -> Submission:
    backend = _ionq_backend(token, name)
    # The SDK sizes the backend with an online lookup and silently falls back to 4
    # qubits when it fails, which would wrongly reject our 5-9 qubit circuits. IonQ
    # is all-to-all connected, so translate to its gate set without a device target.
    # Optimization level 0 matches the notebook's IonQ run.
    tc = transpile(circuit, basis_gates=ge.IONQ_BASIS, optimization_level=0,
                   seed_transpiler=1234)
    job = backend.run(tc, shots=shots)
    return Submission("IonQ", backend.name, job.job_id(), shots, ge.circuit_metrics(tc))


def _ionq_fetch(job_id, token, name):
    job = _ionq_backend(token, name).retrieve_job(job_id)
    status = normalize_status(job.status())
    if status == "done":
        return status, dict(job.result().get_counts()), None
    err = None
    if status == "error":
        try:
            err = str(job.status().value)
        except Exception:
            err = "The job failed on the IonQ side."
    return status, None, err


# ---------------------------------------------------------------- public API
def submit(provider: str, circuit: QuantumCircuit, shots: int, *, token: str,
           crn: str | None = None, backend_name: str | None = None,
           iqm_url: str | None = None) -> Submission:
    """Transpile for the chosen QPU and queue the job. Returns immediately."""
    if not token or not token.strip():
        raise HardwareError("Please enter an API key.")
    token = token.strip()
    try:
        if provider == "IBM Quantum":
            if not crn or not crn.strip():
                raise HardwareError("Please enter the instance CRN.")
            if not backend_name:
                raise HardwareError("Please connect and choose an IBM backend first.")
            return _ibm_submit(circuit, shots, token, crn.strip(), backend_name)
        if provider == "IQM":
            if not backend_name:
                raise HardwareError("Please choose an IQM quantum computer.")
            return _iqm_submit(circuit, shots, token, iqm_url, backend_name.strip())
        if provider == "IonQ":
            return _ionq_submit(circuit, shots, token, backend_name)
        raise HardwareError(f"Unknown provider: {provider}")
    except HardwareError:
        raise
    except Exception as exc:
        raise _wrap(exc, token, crn) from None


def fetch(provider: str, job_id: str, *, token: str, crn: str | None = None,
          backend_name: str | None = None, iqm_url: str | None = None,
          creg: str = "c"):
    """Poll a job. Returns (status, counts_or_None, error_message_or_None)."""
    if not token or not token.strip():
        raise HardwareError("Please enter the API key to check this job.")
    token = token.strip()
    try:
        if provider == "IBM Quantum":
            return _ibm_fetch(job_id, token, (crn or "").strip(), creg)
        if provider == "IQM":
            return _iqm_fetch(job_id, token, iqm_url, (backend_name or "").strip())
        if provider == "IonQ":
            return _ionq_fetch(job_id, token, backend_name)
        raise HardwareError(f"Unknown provider: {provider}")
    except HardwareError:
        raise
    except Exception as exc:
        raise _wrap(exc, token, crn) from None
