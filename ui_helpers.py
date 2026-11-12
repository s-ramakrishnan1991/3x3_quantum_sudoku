"""Small presentation helpers shared by the app tabs."""
import altair as alt
import pandas as pd
import streamlit as st

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


def inject_css():
    st.markdown(GRID_CSS, unsafe_allow_html=True)


def grid_html(rows=None, fixed=None, check=None):
    """rows: 3x3 values (or None for the puzzle view). fixed: clue dict to highlight."""
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


def fmt_rows(rows):
    return "|".join("".join(map(str, r)) for r in rows) if rows else "invalid code"


def outcome_chart(prob, counts, top_n=12):
    """Horizontal bar chart of the most frequent outcomes, coloured valid/invalid."""
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    df = pd.DataFrame({
        "outcome": [f"{k}  →  {fmt_rows(prob.decode(prob.key_to_bits(k)))}" for k, _ in top],
        "counts": [v for _, v in top],
        "valid": ["valid" if prob.is_valid(prob.decode(prob.key_to_bits(k))) else "invalid"
                  for k, _ in top],
    })
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("counts:Q", title="Shots"),
        y=alt.Y("outcome:N", sort="-x", title=None),
        color=alt.Color("valid:N", scale=alt.Scale(domain=["valid", "invalid"],
                                                   range=["#21c354", "#ff4b4b"]), title=None),
        tooltip=["outcome", "counts", "valid"],
    ).properties(height=320, title=f"Top {top_n} measurement outcomes (bitstring → rows)")
