"""Streamlit UI for the optional real-hardware tab (IBM Quantum, IQM, IonQ)."""
from __future__ import annotations

import time

import streamlit as st

import grover_engine as ge
import real_backends as rb
from ui_helpers import grid_html, outcome_chart

ICON = {"queued": "⏳", "running": "⚙️", "done": "✅", "error": "❌", "cancelled": "🚫"}
POLL_SECONDS = 15


# ------------------------------------------------------------------ state
def _init_state():
    ss = st.session_state
    ss.setdefault("hw_jobs", [])        # submitted jobs (no credentials stored here)
    ss.setdefault("hw_results", {})     # job_id -> finished result
    ss.setdefault("ibm_backends", [])   # list of {"name","qubits","pending_jobs"}


def hardware_records(fixed: dict, iters: int) -> list[dict]:
    """Finished hardware results for the current puzzle (used by the comparison tab)."""
    key = tuple(sorted(fixed.items()))
    return [r for r in st.session_state.get("hw_results", {}).values()
            if r["fixed_items"] == key and r["iters"] == iters]


def _creds(provider: str) -> dict:
    """Read credentials from the (always-rendered) input widgets at call time."""
    ss = st.session_state
    if provider == "IBM Quantum":
        return {"token": ss.get("ibm_token", ""), "crn": ss.get("ibm_crn", "")}
    if provider == "IQM":
        return {"token": ss.get("iqm_token", ""),
                "iqm_url": (ss.get("iqm_url") or rb.IQM_DEFAULT_URL).strip()}
    return {"token": ss.get("ionq_token", "")}


# ------------------------------------------------------------------ polling
def _refresh(job: dict):
    try:
        status, counts, err = rb.fetch(
            job["provider"], job["job_id"], backend_name=job["backend"],
            creg="c", **_creds(job["provider"]))
    except rb.HardwareError as exc:
        job["poll_error"] = str(exc)
        job["polled_at"] = time.time()
        return
    job.update(poll_error=None, status=status, polled_at=time.time())
    if status == "done":
        prob = ge.build_problem(dict(job["fixed_items"]))
        st.session_state.hw_results[job["job_id"]] = {
            **{k: job[k] for k in ("provider", "backend", "job_id", "shots", "metrics",
                                   "fixed_items", "iters")},
            "counts": counts,
            "analysis": ge.analyse_counts(prob, counts),
        }
    elif status in ("error", "cancelled"):
        job["error"] = err or f"Job {status}."


def _render_job(job: dict):
    prob = ge.build_problem(dict(job["fixed_items"]))
    fixed = dict(job["fixed_items"])
    status = job["status"]
    head = f"{ICON[status]} {job['provider']} · {job['backend']} · {status}"
    with st.expander(head, expanded=(status == "done")):
        st.caption(f"Job ID: `{job['job_id']}` · {job['shots']} shots · "
                   f"submitted {time.strftime('%H:%M:%S', time.localtime(job['submitted']))}")
        if job.get("metrics"):
            m = job["metrics"]
            st.caption(f"Transpiled: {m['active qubits']} qubits, depth {m['total depth']}, "
                       f"{m['2Q gates']} two-qubit gates (2Q depth {m['2Q depth']})")
        if job.get("poll_error"):
            st.warning(f"Could not check status: {job['poll_error']}")
        if status in ("error", "cancelled"):
            st.error(job.get("error") or "The job did not complete.")
        res = st.session_state.hw_results.get(job["job_id"])
        if status == "done" and res:
            a = res["analysis"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Success probability", f"{a['success']:.1%}")
            c2.metric("Random-guess baseline", f"{len(prob.marked) / prob.search_space:.1%}")
            c3.metric("Best outcome valid", "Yes" if a["best_valid"] else "No")
            if a["best_rows"]:
                st.markdown(grid_html(a["best_rows"], fixed, "ok" if a["best_valid"] else "bad"),
                            unsafe_allow_html=True)
            st.altair_chart(outcome_chart(prob, res["counts"]), width="stretch")


def _jobs_panel():
    jobs = st.session_state.hw_jobs
    if not jobs:
        st.info("No hardware jobs yet. Submit one from a provider tab above.")
        return
    manual = st.button("🔄 Refresh status now", key="hw_refresh")
    newly_final = False
    for job in jobs:
        if job["status"] in rb.FINAL_STATES:
            continue
        stale = time.time() - job.get("polled_at", 0) >= POLL_SECONDS - 1
        if manual or stale:
            before = job["status"]
            _refresh(job)
            newly_final |= job["status"] != before and job["status"] in rb.FINAL_STATES
    for job in reversed(jobs):
        _render_job(job)
    if newly_final:
        st.rerun()  # update the comparison tab and stop auto-polling


def _render_jobs():
    active = any(j["status"] not in rb.FINAL_STATES for j in st.session_state.hw_jobs)
    # Poll automatically only while something is still queued/running.
    st.fragment(run_every=POLL_SECONDS if active else None)(_jobs_panel)()


# ------------------------------------------------------------------ submit
def _submit(provider, fixed, prob, shots, iters, **kwargs):
    circuit = ge.build_grover(prob, iters)
    with st.spinner(f"Transpiling and submitting to {provider}…"):
        try:
            sub = rb.submit(provider, circuit, shots, **kwargs)
        except rb.HardwareError as exc:
            st.error(f"Submission failed: {exc}")
            return
    st.session_state.hw_jobs.append({
        "provider": sub.provider, "backend": sub.backend, "job_id": sub.job_id,
        "shots": sub.shots, "metrics": sub.metrics,
        "fixed_items": tuple(sorted(fixed.items())), "iters": iters,
        "status": "queued", "submitted": time.time(), "polled_at": 0,
    })
    st.toast(f"Job submitted to {provider}: {sub.job_id}")
    st.rerun()


# ------------------------------------------------------------------ main UI
def render(fixed: dict, prob: ge.Problem, shots: int, iters: int):
    _init_state()
    st.subheader("Run on real quantum hardware (optional)")
    n_tot = prob.num_qubits + max(0, prob.num_qubits - 3)
    st.markdown(
        f"The current puzzle uses a **{n_tot}-qubit** circuit ({prob.num_qubits} search + "
        f"{n_tot - prob.num_qubits} ancilla), **{iters}** Grover iterations and **{shots}** shots. "
        "Real devices are noisy, so expect much lower success probability than on the simulators, "
        "especially for medium puzzles."
    )
    st.warning(
        "Hardware jobs use **your own** account and can consume quota or credits (IonQ and paid IBM "
        "plans can incur charges). Jobs queue for minutes to hours; you can leave this page open "
        "and the status updates automatically. API keys are used only for this browser session, "
        "are never stored or logged by the app, and are scrubbed from error messages."
    )
    ack = st.checkbox("I understand this will run on real hardware using my account.", key="hw_ack")

    tab_ibm, tab_iqm, tab_ionq = st.tabs(["IBM Quantum", "IQM", "IonQ"])

    # ---- IBM
    with tab_ibm:
        st.text_input("API key", type="password", key="ibm_token",
                      help="IBM Quantum Platform → API key")
        st.text_input("Instance CRN", key="ibm_crn",
                      placeholder="crn:v1:bluemix:public:quantum-computing:...",
                      help="Cloud Resource Name of your IBM Quantum instance.")
        if st.button("Connect & list backends", key="ibm_connect"):
            token, crn = st.session_state.ibm_token, st.session_state.ibm_crn
            if not token.strip() or not crn.strip():
                st.error("Enter both the API key and the CRN.")
            else:
                with st.spinner("Contacting IBM Quantum…"):
                    try:
                        st.session_state.ibm_backends = rb.ibm_list_backends(token.strip(), crn.strip())
                        if not st.session_state.ibm_backends:
                            st.warning("Connected, but no operational QPUs are available for this instance.")
                    except rb.HardwareError as exc:
                        st.session_state.ibm_backends = []
                        st.error(f"Could not connect: {exc}")
        backends = st.session_state.ibm_backends
        ibm_choice = None
        if backends:
            labels = {b["name"]: f"{b['name']} · {b['qubits']} qubits · "
                                 f"{'?' if b['pending_jobs'] is None else b['pending_jobs']} pending jobs"
                      for b in backends}
            ibm_choice = st.selectbox("Backend", list(labels), format_func=labels.get, key="ibm_backend")
        else:
            st.caption("Press **Connect & list backends** to choose a QPU.")
        if st.button("🚀 Submit to IBM Quantum", key="ibm_submit", type="primary",
                     disabled=not (ack and ibm_choice)):
            _submit("IBM Quantum", fixed, prob, shots, iters, token=st.session_state.ibm_token,
                    crn=st.session_state.ibm_crn, backend_name=ibm_choice)

    # ---- IQM
    with tab_iqm:
        st.text_input("API key", type="password", key="iqm_token",
                      help="IQM Resonance API token")
        choice = st.selectbox("Quantum computer", rb.IQM_COMPUTERS + ["Other…"], key="iqm_computer")
        computer = st.text_input("Quantum computer alias", key="iqm_custom") if choice == "Other…" else choice
        with st.expander("Advanced"):
            st.text_input("IQM server URL", value=rb.IQM_DEFAULT_URL, key="iqm_url")
        if st.button("🚀 Submit to IQM", key="iqm_submit", type="primary",
                     disabled=not (ack and computer)):
            _submit("IQM", fixed, prob, shots, iters, token=st.session_state.iqm_token,
                    backend_name=computer, iqm_url=_creds("IQM")["iqm_url"])

    # ---- IonQ
    with tab_ionq:
        st.text_input("API key", type="password", key="ionq_token", help="IonQ Cloud API key")
        with st.expander("Advanced"):
            st.text_input("QPU name", value=rb.IONQ_BACKEND, key="ionq_backend",
                          help="Only change this if the generic target is rejected, "
                               "e.g. ionq_qpu.forte-1")
        if st.button("🚀 Submit to IonQ", key="ionq_submit", type="primary", disabled=not ack):
            _submit("IonQ", fixed, prob, shots, iters, token=st.session_state.ionq_token,
                    backend_name=(st.session_state.ionq_backend or rb.IONQ_BACKEND).strip())

    # ---- jobs
    st.markdown("---")
    st.subheader("Hardware jobs")
    _render_jobs()

    with st.expander("Retrieve an earlier job by ID"):
        st.caption("Use this if you closed the page while a job was queued. The result is decoded "
                   "against the **puzzle currently shown**, so make sure it is the same puzzle "
                   "and iteration count you submitted. Enter the API key (and CRN for IBM) above first.")
        r_provider = st.selectbox("Provider", rb.PROVIDERS, key="hw_ret_provider")
        r_id = st.text_input("Job ID", key="hw_ret_id")
        r_backend = st.text_input("Backend / quantum computer name", key="hw_ret_backend",
                                  help="Needed for IQM (alias) and IonQ (QPU name). Optional for IBM.")
        if st.button("Retrieve", key="hw_ret_btn", disabled=not r_id.strip()):
            if any(j["job_id"] == r_id.strip() for j in st.session_state.hw_jobs):
                st.info("That job is already in the list above.")
            else:
                st.session_state.hw_jobs.append({
                    "provider": r_provider,
                    "backend": r_backend.strip() or ("ionq_qpu" if r_provider == "IonQ" else "unknown"),
                    "job_id": r_id.strip(), "shots": shots, "metrics": None,
                    "fixed_items": tuple(sorted(fixed.items())), "iters": iters,
                    "status": "queued", "submitted": time.time(), "polled_at": 0,
                })
                st.rerun()
