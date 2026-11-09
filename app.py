"""Interactive 3x3 Sudoku solver with Grover's algorithm + backend comparison study."""
import math

import altair as alt
import pandas as pd
import streamlit as st

import grover_engine as ge

st.set_page_config(page_title="Quantum Sudoku · Grover", page_icon="🧩", layout="wide")

# ---------------------------------------------------------------- helpers
GRID_CSS = """
<style>
.sgrid{border-collapse:collapse;margin:0.25rem 0 0.5rem 0}
.sgrid td{width:3.2rem;height:3.2rem;text-align:center;font-size:1.5rem;font-weight:600;
          border:1.5px solid rgba(128,128,128,.55)}
.sgrid td.clue{background:rgba(255,75,75,.18)}
.sgrid td.empty{color:rgba(128,128,128,.6)}
.sgrid td.ok{background:rgba(33,195,84,.20)}
.sgrid td.bad{background:rgba(255,75,75,.25)}
</style>
"""
st.markdown(GRID_CSS, unsafe_allow_html=True)


def grid_html(rows=None, fixed=None, check=None):
    """rows: 3x3 values (or None for puzzle view). fixed: clue dict highlighting."""
    fixed = fixed or {}
    out = ['<table class="sgrid">']
    for r in range(3):
        out.append("<tr>")
        for c in range(3):
            if rows is None:
                v = fixed.get((r, c))
                cls = "clue" if v is not None else "empty"
                txt = str(v) if v is not None else "·"
            else:
                txt = str(rows[r][c])
                cls = "clue" if (r, c) in fixed else (check or "")
            out.append(f'<td class="{cls}">{txt}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


@st.cache_data(show_spinner=False)
def run_cached(fixed_items: tuple, backend: str, shots: int, iterations: int):
    fixed = dict(fixed_items)
    prob = ge.build_problem(fixed)
    qc = ge.build_grover(prob, iterations)
    tc, counts, secs = ge.run_backend(backend, qc, shots)
    m = ge.circuit_metrics(tc)
    a = ge.analyse_counts(prob, counts)
    return m, counts, secs, a


def fmt_rows(rows):
    return "|".join("".join(map(str, r)) for r in rows) if rows else "invalid code"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧩 Quantum Sudoku")
difficulty = st.sidebar.radio(
    "Difficulty", ["easy", "medium"], horizontal=True,
    help="Easy: 3 clues on the diagonal. Medium: 2 clues. Fewer clues → more qubits → deeper circuits.",
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

tab_solve, tab_compare, tab_how, tab_notes = st.tabs(
    ["🎮 Solve", "📊 Backend comparison", "🔬 How it works", "ℹ️ Notes"]
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

            top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
            df = pd.DataFrame({
                "outcome": [f"{k}  →  {fmt_rows(prob.decode(prob.key_to_bits(k)))}" for k, _ in top],
                "counts": [v for _, v in top],
                "valid": ["valid" if prob.is_valid(prob.decode(prob.key_to_bits(k))) else "invalid"
                          for k, _ in top],
            })
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("counts:Q", title="Shots"),
                y=alt.Y("outcome:N", sort="-x", title=None),
                color=alt.Color("valid:N", scale=alt.Scale(domain=["valid", "invalid"],
                                                           range=["#21c354", "#ff4b4b"]), title=None),
                tooltip=["outcome", "counts", "valid"],
            ).properties(height=320, title="Top 12 measurement outcomes (bitstring → rows)")
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
        df = pd.DataFrame(cmp_[1])
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
                color=alt.Color("Noise:N", scale=alt.Scale(domain=["Noiseless", "Noisy"],
                                                           range=["#4c78a8", "#f58518"])),
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
        with st.expander("Gate breakdown per backend"):
            st.dataframe(pd.DataFrame(cmp_[2]).fillna(0).astype(int).T, width="stretch")
    else:
        st.info("Select backends and press **Run comparison**. Try both difficulty levels: the medium "
                "puzzle needs more qubits and a deeper circuit, so noisy backends degrade much faster.")

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
The notebook uses the IonQ cloud simulator (API key), an IQM fake backend (needs Qiskit 1.x,
incompatible with the IBM stack), and optionally a real IBM QPU (account and queue time). None
of these can run in a shared public deployment, so this app uses credential-free equivalents
that preserve the comparison:

| Backend here | Stands in for | What is preserved |
|---|---|---|
| Aer ideal | Notebook's local Aer run | Identical: `rz, sx, x, cx`, opt. level 3 |
| IonQ-style ideal | IonQ simulator | Same gate set and opt. level 0; ideal simulation on Aer |
| IQM-style noisy | IQM FakeApollo | 20-qubit square lattice, native `r, cz`; **synthetic** noise, not IQM's calibration |
| IBM Marrakesh noisy | IBM fake backend | Real `ibm_marrakesh` snapshot (heavy-hex, calibration noise) |

**Dropped:** the *hard* and *empty* difficulties (as requested), real-QPU execution, and M3
readout mitigation (needs the `mthree` compiled package and a real backend's calibration
circuits). Because of this, results on the noisy backends are a fair relative comparison but
absolute numbers will differ from the IQM hardware model.

Clue values are random but always distinct, which guarantees at least one valid solution.
        """
    )
