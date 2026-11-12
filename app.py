"""Interactive 3x3 Sudoku solver with Grover's algorithm + backend comparison study."""
import math

import altair as alt
import pandas as pd
import streamlit as st

import grover_engine as ge
import hardware_tab
from ui_helpers import fmt_rows, grid_html, inject_css, outcome_chart

st.set_page_config(page_title="Quantum Sudoku · Grover", page_icon="🧩", layout="wide")
inject_css()


@st.cache_data(show_spinner=False)
def run_cached(fixed_items: tuple, backend: str, shots: int, iterations: int):
    fixed = dict(fixed_items)
    prob = ge.build_problem(fixed)
    qc = ge.build_grover(prob, iterations)
    tc, counts, secs = ge.run_backend(backend, qc, shots)
    m = ge.circuit_metrics(tc)
    a = ge.analyse_counts(prob, counts)
    return m, counts, secs, a


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧩 Quantum Sudoku")
difficulty = st.sidebar.radio(
    "Difficulty", ["easy", "medium"], horizontal=True,
    help="Easy: 3 clues. Medium: 2 clues. Clue positions are random anywhere on the grid. Fewer clues → more qubits → deeper circuits.",
)
if "seed" not in st.session_state:
    st.session_state.seed = 1
if st.sidebar.button("🎲 New random puzzle", width="stretch"):
    st.session_state.seed += 1

shots = st.sidebar.select_slider("Shots", [512, 1024, 2048, 4096, 8192], value=2048)

fixed = ge.make_puzzle(difficulty, st.session_state.seed)
prob = ge.build_problem(fixed)
opt_iters = ge.grover_iterations(prob.search_space, len(prob.marked))

with st.sidebar.expander("Advanced"):
    override = st.checkbox("Override Grover iterations", value=False)
    iters = st.slider("Iterations", 1, 3 * opt_iters + 2, opt_iters, disabled=not override)
iters = iters if override else opt_iters

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Puzzle:** {difficulty} · seed {st.session_state.seed}\n\n"
    f"**Search qubits:** {prob.num_qubits} · **Search space:** {prob.search_space}\n\n"
    f"**Valid solutions:** {len(prob.marked)} · **Iterations:** {iters} "
    f"(optimal {opt_iters})"
)

# ---------------------------------------------------------------- main
st.title("Solving a 3×3 Sudoku with Grover's algorithm")
st.write(
    "Fill a 3×3 grid with the values 0, 1, 2 so that every row and column contains each "
    "value exactly once. A quantum search finds the solution, and you can compare how "
    "different quantum backends cope with the same circuit."
)

tab_solve, tab_compare, tab_hw, tab_how, tab_notes = st.tabs(
    ["🎮 Solve", "📊 Backend comparison", "☁️ Real hardware", "🔬 How it works", "ℹ️ Notes"]
)

# ================================================================= SOLVE
with tab_solve:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Puzzle")
        st.markdown(grid_html(None, fixed), unsafe_allow_html=True)
        st.caption("Highlighted cells are fixed clues; · is empty.")
        with st.expander("Reveal all valid solutions (classical check)"):
            for i, bits in enumerate(prob.marked, 1):
                st.markdown(f"**Solution {i}**")
                st.markdown(grid_html(prob.decode(bits), fixed, "ok"), unsafe_allow_html=True)
    with right:
        st.subheader("Run Grover")
        solve_backend = st.selectbox("Backend", list(ge.BACKENDS), key="solve_backend")
        info = ge.BACKENDS[solve_backend]
        st.caption(f"{info['vendor']} · {info['noise']} · native gates: {info['native']}. {info['desc']}")
        if st.button("▶ Run on this backend", type="primary"):
            with st.spinner(f"Running on {solve_backend}… (first IBM run takes ~15 s)"):
                st.session_state.solve_result = (
                    (tuple(sorted(fixed.items())), solve_backend, shots, iters),
                    run_cached(tuple(sorted(fixed.items())), solve_backend, shots, iters),
                )
        res = st.session_state.get("solve_result")
        if res and res[0] == (tuple(sorted(fixed.items())), solve_backend, shots, iters):
            m, counts, secs, a = res[1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Success probability", f"{a['success']:.1%}",
                      help="Share of shots that decode to a valid Sudoku solution.")
            c2.metric("Random-guess baseline", f"{len(prob.marked) / prob.search_space:.1%}")
            c3.metric("2-qubit depth", m["2Q depth"])
            if a["best_valid"]:
                st.success("Most frequent outcome is a valid solution.")
            else:
                st.error("Most frequent outcome is NOT a valid solution. Noise has swamped the signal.")
            st.markdown("**Most frequent outcome, decoded**")
            st.markdown(grid_html(a["best_rows"], fixed, "ok" if a["best_valid"] else "bad")
                        if a["best_rows"] else "_Outcome is not a legal row encoding._",
                        unsafe_allow_html=True)

            chart = outcome_chart(prob, counts)
            st.altair_chart(chart, width="stretch")
        else:
            st.info("Pick a backend and press **Run**.")

# ================================================================= COMPARE
with tab_compare:
    st.subheader("Backend comparison on the current puzzle")
    st.write(
        "The *same* logical Grover circuit is transpiled for each backend's native gate set and "
        "connectivity, then simulated. Transpiled depth and two-qubit gate count drive how much noise "
        "accumulates."
    )
    chosen = st.multiselect("Backends to compare", list(ge.BACKENDS), default=list(ge.BACKENDS))
    if st.button("▶ Run comparison", type="primary", disabled=not chosen):
        rows, ops_rows = [], {}
        prog = st.progress(0.0)
        for i, b in enumerate(chosen):
            prog.progress(i / len(chosen), text=f"Running {b}…")
            m, counts, secs, a = run_cached(tuple(sorted(fixed.items())), b, shots, iters)
            rows.append({
                "Backend": b,
                "Vendor": ge.BACKENDS[b]["vendor"],
                "Noise": ge.BACKENDS[b]["noise"],
                "Native gates": ge.BACKENDS[b]["native"],
                "Register width": m["register width"],
                "Active qubits": m["active qubits"],
                "Shots": shots,
                "Total depth": m["total depth"],
                "2Q depth": m["2Q depth"],
                "2Q gates": m["2Q gates"],
                "Total gates": m["total gates"],
                "Success prob.": a["success"],
                "Best outcome valid": "✅" if a["best_valid"] else "❌",
                "Run time (s)": round(secs, 1),
            })
            ops_rows[b] = m["ops"]
        prog.empty()
        st.session_state.compare = (
            (tuple(sorted(fixed.items())), tuple(chosen), shots, iters), rows, ops_rows)

    cmp_ = st.session_state.get("compare")
    if cmp_ and cmp_[0] == (tuple(sorted(fixed.items())), tuple(chosen), shots, iters):
        hw_rows = []
        for r in hardware_tab.hardware_records(fixed, iters):
            m = r["metrics"] or {}
            hw_rows.append({
                "Backend": f"{r['provider']} · {r['backend']}",
                "Vendor": r["provider"], "Noise": "Real hardware", "Native gates": "device native",
                "Register width": m.get("register width"), "Active qubits": m.get("active qubits"),
                "Shots": r["analysis"]["shots"], "Total depth": m.get("total depth"),
                "2Q depth": m.get("2Q depth"), "2Q gates": m.get("2Q gates"),
                "Total gates": m.get("total gates"), "Success prob.": r["analysis"]["success"],
                "Best outcome valid": "✅" if r["analysis"]["best_valid"] else "❌",
                "Run time (s)": None,
            })
        df = pd.DataFrame(cmp_[1] + hw_rows)
        int_cols = ["Register width", "Active qubits", "Shots", "Total depth", "2Q depth",
                    "2Q gates", "Total gates"]
        df[int_cols] = df[int_cols].astype("Int64")
        if hw_rows:
            st.caption(f"Includes {len(hw_rows)} real-hardware result(s) for this puzzle from the "
                       "☁️ Real hardware tab.")
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={
                "Success prob.": st.column_config.ProgressColumn(
                    "Success prob.", format="percent", min_value=0.0, max_value=1.0),
            },
        )
        baseline = len(prob.marked) / prob.search_space
        c1, c2 = st.columns(2)
        with c1:
            ch = alt.Chart(df).mark_bar().encode(
                x=alt.X("Backend:N", sort=None, axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("Success prob.:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Noise:N", scale=alt.Scale(domain=["Noiseless", "Noisy", "Real hardware"],
                                                           range=["#4c78a8", "#f58518", "#9467bd"])),
                tooltip=["Backend", alt.Tooltip("Success prob.:Q", format=".1%"), "Noise"],
            )
            rule = alt.Chart(pd.DataFrame({"y": [baseline]})).mark_rule(
                strokeDash=[6, 4], color="gray").encode(y="y:Q")
            st.altair_chart((ch + rule).properties(
                height=320, title="Success probability (dashed = random-guess baseline)"),
                width="stretch")
        with c2:
            long = df.melt(id_vars="Backend", value_vars=["Total depth", "2Q depth", "2Q gates"],
                           var_name="Metric", value_name="Value")
            ch2 = alt.Chart(long).mark_bar().encode(
                x=alt.X("Backend:N", sort=None, axis=alt.Axis(labelAngle=-20), title=None),
                xOffset="Metric:N",
                y=alt.Y("Value:Q", title="Transpiled circuit size"),
                color="Metric:N",
                tooltip=["Backend", "Metric", "Value"],
            )
            st.altair_chart(ch2.properties(height=320, title="Circuit cost after transpilation"),
                            width="stretch")

        best = df.loc[df["Success prob."].idxmax()]
        cheapest = df.loc[df["2Q depth"].idxmin()]
        st.markdown(
            f"- **Highest success probability:** {best['Backend']} ({best['Success prob.']:.1%}; "
            f"random guessing would give {baseline:.1%}).\n"
            f"- **Shallowest two-qubit depth:** {cheapest['Backend']} ({cheapest['2Q depth']}).\n"
            f"- Noisy backends lose success probability roughly in step with their 2-qubit depth."
        )
        with st.expander("Gate breakdown per simulated backend"):
            st.dataframe(pd.DataFrame(cmp_[2]).fillna(0).astype(int).T, width="stretch")
    else:
        st.info("Select backends and press **Run comparison**. Try both difficulty levels: the medium "
                "puzzle needs more qubits and a deeper circuit, so noisy backends degrade much faster. "
                "Finished real-hardware jobs for this puzzle are added to the table automatically.")

# ================================================================= HARDWARE
with tab_hw:
    hardware_tab.render(fixed, prob, shots, iters)

# ================================================================= HOW
with tab_how:
    st.subheader("How the puzzle becomes a quantum circuit")
    st.markdown(
        """
1. **Row-permutation encoding.** Every row of a valid grid is a permutation of 0, 1, 2, so row
   constraints are satisfied by construction. Each row is encoded as an index into its list of
   legal permutations (after applying the clues), which needs only ⌈log₂ n⌉ qubits per row.
2. **Oracle.** A phase flip is applied to every basis state whose rows also satisfy the
   *column* constraints (computed classically). Each flip is a multi-controlled Z built with
   V-chain MCX gates and clean ancillas to keep depth low.
3. **Diffuser.** Standard inversion about the mean.
4. **Iterations.** The optimal count is ≈ π/4 · √(N/M), with N the search space and M the
   number of valid solutions.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        rows_info = pd.DataFrame({
            "Row": [0, 1, 2],
            "Legal permutations": [len(c) for c in prob.row_choices],
            "Qubits": prob.row_widths,
        })
        st.dataframe(rows_info, hide_index=True, width="stretch")
        st.caption(f"Total: {prob.num_qubits} search qubits → N = {prob.search_space}, "
                   f"M = {len(prob.marked)}, optimal iterations = {opt_iters}.")
    with c2:
        theta = math.asin(math.sqrt(len(prob.marked) / prob.search_space))
        ks = list(range(0, 3 * opt_iters + 3))
        curve = pd.DataFrame({"iterations": ks,
                              "P(valid solution)": [math.sin((2 * k + 1) * theta) ** 2 for k in ks]})
        line = alt.Chart(curve).mark_line(point=True).encode(
            x="iterations:Q", y=alt.Y("P(valid solution):Q", scale=alt.Scale(domain=[0, 1])))
        mark = alt.Chart(pd.DataFrame({"x": [iters]})).mark_rule(color="#ff4b4b").encode(x="x:Q")
        st.altair_chart((line + mark).properties(
            height=240, title="Ideal success probability vs. Grover iterations (red = current)"),
            width="stretch")
    with st.expander("Show the logical circuit (1 Grover iteration)"):
        try:
            fig = ge.build_grover(prob, 1).draw("mpl", fold=40, scale=0.6)
            st.pyplot(fig)
        except Exception as exc:  # drawing is cosmetic only
            st.warning(f"Could not draw circuit: {exc}")

# ================================================================= NOTES
with tab_notes:
    st.subheader("Backends and differences from the notebook")
    st.markdown(
        """
**Simulated backends** (no credentials, used on the Solve and comparison tabs):

| Backend here | Stands in for | What is preserved |
|---|---|---|
| Aer ideal | Notebook's local Aer run | Identical: `rz, sx, x, cx`, opt. level 3 |
| IonQ-style ideal | IonQ simulator | Same gate set and opt. level 0; ideal simulation on Aer |
| IQM-style noisy | IQM FakeApollo | 20-qubit square lattice, native `r, cz`; **synthetic** noise, not IQM's calibration |
| IBM Marrakesh noisy | IBM fake backend | Real `ibm_marrakesh` snapshot (heavy-hex, calibration noise) |

**Real hardware** (optional, ☁️ tab): IBM Quantum (API key + CRN + choice of QPU), IQM (API key +
choice of quantum computer) and IonQ (API key). Jobs run on your own account, are queued
asynchronously, and finished results are added to the comparison table. Keys live only in your
browser session's server memory and are never stored, cached or logged.

**Puzzles:** clues are placed at random cells of the grid and always taken from a valid solved
grid, so every puzzle is solvable. Easy has 3 clues (at most 4 search qubits), medium has 2 clues
(5-6 search qubits). Some puzzles have more than one valid solution; any of them counts as success,
and the Grover iteration count is chosen to maximise the ideal success probability.

**Dropped from the notebook:** the *hard* and *empty* difficulties (as requested) and M3 readout
mitigation (needs the compiled `mthree` package and a real backend's calibration circuits).
        """
    )
