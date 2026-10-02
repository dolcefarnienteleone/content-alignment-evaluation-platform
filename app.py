"""
Content Alignment Evaluation Platform — Streamlit demo (P0-C)

Reads ONLY the static snapshot in data/demo/ (built by scripts/build_demo_dataset.py).
No API keys, no live LLM, no local paths.

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data" / "demo"
REPO_URL = "https://github.com/dolcefarnienteleone/content-alignment-evaluation-platform"
AUTHOR = "" #TODO(P0-D): add author name

st.set_page_config(
    page_title="Content Alignment Evaluation Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@st.cache_data
def load():
    def csv(name):
        return pd.read_csv(DATA / name)
    return {
        "content": csv("demo_content.csv").sort_values("demo_order"),
        "insights": csv("demo_decision_insights.csv"),
        "pairs": csv("demo_alignment_pairs.csv"),
        "evidence": csv("demo_alignment_evidence.csv"),
        "profiles": csv("demo_concept_profiles.csv"),
        "rules": csv("demo_relationship_rules.csv"),
        "metrics": csv("demo_system_metrics.csv"),
        "manifest": json.loads((DATA / "demo_manifest.json").read_text(encoding="utf-8")),
    }


D = load()

COMPARISONS = [
    (
        "intent_to_content",
        "Framing → Content",
        "Was the inferred framing realized in the content?",
    ),
    (
        "content_to_audience",
        "Content → Audience",
        "Which content signals carried through to audience perception?",
    ),
    (
        "intent_to_audience",
        "Framing → Audience",
        "Which inferred framing signals carried through end-to-end?",
    ),
]
STATUS = {
    "aligned_concept": ("🟢", "Aligned"),
    "partial_alignment": ("🟡", "Partial"),
    "missing_resonance": ("⚪", "No governed counterpart"),
}
LAYER_LABEL = {
    "intent": "Inferred framing",
    "content": "Content understanding",
    "audience": "Audience perception",
}
REL_EXPLAIN = {
    "exact": "same canonical concept on both sides",
    "contributes_to": "source concept contributes to the target-side perception",
    "manifests_as": "source concept shows up as a different target-side perception",
    "related_to": "related concepts (weaker link)",
}


def pct(x, nd=0):
    return "—" if pd.isna(x) else f"{x:.{nd}%}"


def score_fmt(x):
    return "—" if pd.isna(x) else f"{x:.2f}"


def metrics_section(section):
    m = D["metrics"]
    return m[m.section == section]


# ---------------------------------------------------------------------------
# P0-C: path diagram + evidence-at-a-glance helpers
# ---------------------------------------------------------------------------

# Theme-neutral styling: translucent fills + inherited text color, so it reads in light and dark mode.
st.markdown(
    """
<style>
.ae-path { display:grid; grid-template-columns: minmax(0,1fr) 150px minmax(0,1fr) 150px minmax(0,1fr);
           gap:10px 6px; align-items:center; margin: 6px 0 4px; }
.ae-colhead { font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; opacity:.6; }
.ae-node { border:1px solid rgba(128,128,128,.35); border-radius:10px; padding:12px 14px;
           background: rgba(128,128,128,.06); font-weight:600; line-height:1.3; }
.ae-node.ok   { border-color: rgba(46,160,67,.65); background: rgba(46,160,67,.10); }
.ae-node.none { border-style:dashed; font-weight:400; font-style:italic; opacity:.75; background:transparent; }
.ae-edge { text-align:center; font-size:.78rem; line-height:1.25; }
.ae-edge .arrow { font-size:1.3rem; display:block; line-height:1; }
.ae-edge .rel { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.ae-edge.none { opacity:.6; }
.ae-lbl { display:none; }
@media (max-width: 720px) {
  .ae-path { grid-template-columns: 1fr; }
  .ae-colhead { display:none; }
  .ae-lbl { display:block; font-size:.68rem; letter-spacing:.06em; text-transform:uppercase; opacity:.6; font-weight:400; font-style:normal; margin-bottom:2px; }
  .ae-edge .arrow { transform: rotate(90deg); display:inline-block; margin: 2px 0; }
}
.ae-quote { border-left:3px solid rgba(128,128,128,.4); padding:2px 0 2px 12px; margin:6px 0 2px; }
.ae-quote.ok { border-left-color: rgba(46,160,67,.7); }
.ae-meta { font-size:.8rem; opacity:.65; margin:0 0 12px 15px; }
</style>
""",
    unsafe_allow_html=True,
)


def _esc(s) -> str:
    import html
    return html.escape(str(s)) if isinstance(s, str) else "—"


def focus_chains(pairs_df: pd.DataFrame) -> list[dict]:
    """Framing → Content → Audience chains for the case's focus concepts."""
    f = pairs_df[pairs_df.is_focus_pair]
    ic = f[(f.comparison_type == "intent_to_content") & (f.insight_type != "missing_resonance")]
    ca = f[f.comparison_type == "content_to_audience"]
    chains = []
    for r in ic.itertuples():
        nxt = ca[ca.source_concept == r.target_concept]
        n = nxt.iloc[0] if len(nxt) else None
        matched = n is not None and n.insight_type != "missing_resonance"
        chains.append({
            "framing": r.source_concept,
            "content": r.target_concept,
            "edge1": (r.relationship_type, r.relationship_strength),
            "audience": n.target_concept if matched else None,
            "edge2": (n.relationship_type, n.relationship_strength) if matched else None,
        })
    return chains


def render_path(chains: list[dict]) -> None:
    def edge(e):
        if e is None:
            return ('<div class="ae-edge none"><span class="arrow">⇢</span>'
                    '<span>no approved rule</span></div>')
        rel, strength = e
        s = f" · {strength:.2f}" if pd.notna(strength) else ""
        return (f'<div class="ae-edge"><span class="arrow">→</span>'
                f'<span class="rel">{_esc(rel)}</span>{s}</div>')

    cells = ['<div class="ae-colhead">Inferred framing</div><div></div>'
             '<div class="ae-colhead">Content understanding</div><div></div>'
             '<div class="ae-colhead">Audience perception</div>']
    for c in chains:
        L = lambda t: f'<span class="ae-lbl">{t}</span>'
        aud = (f'<div class="ae-node ok">{L("Audience perception")}{_esc(c["audience"])}</div>' if c["audience"]
               else f'<div class="ae-node none">{L("Audience perception")}No governed audience counterpart</div>')
        cells.append(
            f'<div class="ae-node">{L("Inferred framing")}{_esc(c["framing"])}</div>{edge(c["edge1"])}'
            f'<div class="ae-node">{L("Content understanding")}{_esc(c["content"])}</div>{edge(c["edge2"])}{aud}'
        )
    st.markdown(f'<div class="ae-path">{"".join(cells)}</div>', unsafe_allow_html=True)


def quote(text, meta, ok=False) -> None:
    st.markdown(
        f'<div class="ae-quote{" ok" if ok else ""}">{_esc(text)}</div><div class="ae-meta">{meta}</div>',
        unsafe_allow_html=True,
    )


SOURCE_NOUN = {"intent_to_content": "framing", "content_to_audience": "content", "intent_to_audience": "framing"}
TARGET_NOUN = {"intent_to_content": "content", "content_to_audience": "audience", "intent_to_audience": "audience"}


def comparison_fact(pairs_df: pd.DataFrame, key: str) -> str:
    """Plain-language reading of one diagnostic score, computed from the pair table."""
    sub = pairs_df[pairs_df.comparison_type == key]
    if sub.empty:
        return "No source concepts for this comparison."
    matched = sub[sub.insight_type != "missing_resonance"]
    text = (f"**{len(matched)} of {len(sub)}** {SOURCE_NOUN[key]} concepts have a governed "
            f"{TARGET_NOUN[key]} counterpart.")
    if TARGET_NOUN[key] == "audience" and not matched.empty:
        top = matched.sort_values("audience_support_rate", ascending=False).iloc[0]
        if pd.notna(top.audience_support_rate):
            text += (f" The strongest, *{top.target_concept}*, appears in "
                     f"{top.audience_support_rate:.0%} of mapped audience signals.")
    return text


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.title("🎯 Content Alignment Evaluation Platform")
st.markdown(
    "**How does creator framing carry through content and audience perception?** "
    "This system analyzes YouTube content and audience comments to compare **inferred framing**, "
    "**content understanding**, and **audience perception**. It maps signals into a "
    "human-governed ontology and shows, with evidence, what carries through, what remains unmatched, "
    "and what else audiences surface."
)
st.caption(
    "LLM extraction → ontology normalization (human-reviewed) → cross-ontology alignment → interpretable insights · "
    "Static snapshot: no login, no API key, no live LLM."
)

# ---------------------------------------------------------------------------
# content selector  (deep-linkable: ?case=hero | gap | relationship)
# ---------------------------------------------------------------------------

content = D["content"]
roles = content.demo_role.tolist()
labels = {r.demo_role: f"{r.demo_order} · {r.demo_label}" for r in content.itertuples()}
default_role = st.query_params.get("case", roles[0])
if default_role not in roles:
    default_role = roles[0]

role = st.radio(
    "Choose a case",
    roles,
    index=roles.index(default_role),
    format_func=lambda r: labels[r],
    horizontal=True,
)
st.query_params["case"] = role

case = content[content.demo_role == role].iloc[0]
cid = case.content_id
ins = D["insights"][D["insights"].content_id == cid]
pairs = D["pairs"][D["pairs"].content_id == cid]
ev = D["evidence"][D["evidence"].content_id == cid]
prof = D["profiles"][D["profiles"].content_id == cid]

st.divider()

tab_overview, tab_journey, tab_audience, tab_system = st.tabs(
    ["Overview", "Alignment Journey", "Audience Intelligence", "Evaluation & System"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tab_overview:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.image(case.thumbnail_url, width="stretch")
        st.markdown(f"**[{case.display_title_en}]({case.url})**")
        if case.original_title != case.display_title_en:
            st.caption(f"Original title: {case.original_title}")
        st.caption(
            f"{case.creator_name} · {case.publish_date} · {int(case.view_count):,} views · "
            f"{int(case.collected_comment_rows):,} comments collected · {int(case.annotated_comment_count):,} annotated"
        )
    with right:
        st.subheader(case.headline)
        st.write(case.story)
        st.info(f"**What this case shows:** {case.what_it_proves}")

    # --- the path, at a glance
    st.markdown("#### The path through the engine")
    chains = focus_chains(pairs)
    if chains:
        render_path(chains)
        st.caption("Links come only from an exact canonical match or a human-approved relationship rule "
                   "(type · strength). Labels on each side don't need to match.")
    else:
        st.markdown(f"**Focus path:** {case.focus_path}")

    # --- evidence at a glance: what the content shows vs. what viewers said
    linked = any(c["audience"] for c in chains)
    focus_aud = [c["audience"] for c in chains if c["audience"]]
    e_left, e_right = st.columns(2, gap="large")
    with e_left:
        st.markdown("**What the content shows**")
        src = ev[(ev.layer.isin(["intent", "content"])) & ev.is_focus_concept].sort_values("layer_order")
        # one quote per layer (framing, then content), skipping a content quote identical to the framing one
        shown, picks = set(), []
        for layer_name in ["intent", "content"]:
            for r in src[src.layer == layer_name].itertuples():
                if r.evidence_span not in shown:
                    picks.append(r); shown.add(r.evidence_span); break
        for r in picks:
            expl = r.model_explanation if isinstance(r.model_explanation, str) else ""
            if expl and sum(ch.isascii() for ch in expl) / len(expl) < 0.8:
                expl = ""  # only show English model explanations here
            quote(r.evidence_span,
                  f"{LAYER_LABEL[r.layer]} → <b>{_esc(r.canonical_concept)}</b>"
                  + (f"<br>{_esc(expl)}" if expl else ""))
    with e_right:
        aud = ev[(ev.layer == "audience") & ev.is_focus_concept]
        if linked:
            st.markdown("**What viewers said**")
            aud = aud[aud.canonical_concept.isin(focus_aud)]
        else:
            st.markdown("**What viewers expressed instead**")
            st.caption("No approved rule links these to the framing/content concept, so they are not "
                       "counted as alignment.")
        top = (aud.dropna(subset=["evidence_likes"]).sort_values("evidence_likes", ascending=False)
                  .drop_duplicates("evidence_text"))
        # one quote per audience concept first, then fill to 3
        picked = pd.concat([top.drop_duplicates("canonical_concept"), top]).drop_duplicates("evidence_id").head(3)
        for r in picked.itertuples():
            quote(r.evidence_text,
                  f"→ <b>{_esc(r.canonical_concept)}</b> · 👍 {int(r.evidence_likes):,}", ok=linked)

    st.markdown("#### Alignment diagnostics")
    cols = st.columns(3)
    for col, (key, label, question) in zip(cols, COMPARISONS):
        col.metric(
            label,
            score_fmt(case[f"{key}_score"]),
            help=f"{question}\n\nCoverage: {pct(case[f'{key}_coverage'])} of source concepts have a governed match.",
        )
        col.caption(question)
        col.markdown(comparison_fact(pairs, key))
    st.caption(
        "Diagnostic scores combine governed concept coverage, relationship strength, confidence, and audience support. "
        "They are not content-quality ratings, model-accuracy metrics, or literal percentages of audience alignment. "
        "V1 intentionally keeps the three comparisons separate rather than blending them into one overall score."
    )

    st.markdown("#### Key findings")
    c1, c2, c3 = st.columns(3)
    for col, section, title in [
        (c1, "alignment", "✅ What carried through"),
        (c2, "gap", "◌ What remained unmatched"),
        (c3, "audience_intelligence", "+ What audiences added"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            sub = ins[ins.display_section == section].sort_values(["field_order", "rank"])
            if sub.empty:
                st.caption("Nothing detected under the governed ontology.")
            for label, grp in sub.groupby("display_label", sort=False):
                items = " · ".join(grp.statement.tolist())
                st.markdown(f"- *{label}:* {items}")

# ---------------------------------------------------------------------------
# Alignment Journey
# ---------------------------------------------------------------------------

with tab_journey:
    if chains:
        render_path(chains)
    st.caption(
        "Each source concept is matched to the target side only through an exact canonical match or a "
        "**human-approved relationship rule**. If no governed match exists, the engine reports it instead of guessing."
    )

    for key, label, question in COMPARISONS:
        sub = pairs[pairs.comparison_type == key].sort_values(["status_order", "evidence_strength"],
                                                               ascending=[True, False])
        st.markdown(f"#### {label} — score {score_fmt(case[f'{key}_score'])}")
        st.caption(question)
        st.markdown(comparison_fact(pairs, key))
        if sub.empty:
            st.caption("No source concepts for this comparison.")
            continue
        for r in sub.itertuples():
            icon, status = STATUS.get(r.insight_type, ("•", r.insight_type))
            target = r.target_concept if pd.notna(r.target_concept) else "—"
            rel = ""
            if pd.notna(r.relationship_type):
                rel = f" &nbsp; `{r.relationship_type}` · strength {r.relationship_strength:.2f}"
            focus = " ⭐" if r.is_focus_pair else ""
            st.markdown(f"{icon} **{r.source_concept}** → **{target}**{rel} &nbsp; *({status})*{focus}")

    st.divider()
    st.markdown("#### Evidence")
    st.caption("Raw model labels are normalized to canonical concepts. Click a concept to see the quotes behind it.")
    st.caption("Evidence is shown in its original language where available; English summaries are provided "
               "in the Overview for the featured path.")
    concepts = (ev.dropna(subset=["canonical_concept"])
                  .sort_values(["layer_order", "is_focus_concept"], ascending=[True, False]))
    for layer in ["intent", "content", "audience"]:
        lev = concepts[concepts.layer == layer]
        if lev.empty:
            continue
        st.markdown(f"**{LAYER_LABEL[layer]}**")
        for concept, grp in lev.groupby("canonical_concept", sort=False):
            star = "⭐ " if grp.is_focus_concept.any() else ""
            with st.expander(f"{star}{concept}  ({len(grp)})", expanded=bool(grp.is_focus_concept.any())):
                for r in grp.sort_values("evidence_rank").itertuples():
                    if layer == "audience":
                        likes = f" · 👍 {int(r.evidence_likes):,}" if pd.notna(r.evidence_likes) else ""
                        st.markdown(f"> {r.evidence_text}")
                        st.caption(f"model label: *{r.raw_label}*{likes}")
                    else:
                        st.markdown(f"> {r.evidence_span}")
                        expl = f" — {r.model_explanation}" if isinstance(r.model_explanation, str) and r.model_explanation else ""
                        st.caption(f"model label: *{r.raw_label}*{expl}")

    unmapped = ev[(ev.layer != "audience") & (~ev.is_mapped)]
    if not unmapped.empty:
        with st.expander(f"Framing/content labels not in the governed ontology ({len(unmapped)})"):
            st.caption("Kept for transparency but excluded from scoring until a reviewer adds them to the ontology.")
            st.dataframe(unmapped[["layer", "raw_label", "normalization_status", "evidence_span"]],
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Audience Intelligence
# ---------------------------------------------------------------------------

with tab_audience:
    st.caption(
        f"Based on {int(case.annotated_comment_count):,} annotated comments "
        f"(of {int(case.collected_comment_rows):,} collected; {int(case.comment_count):,} on the video)."
    )
    ai = ins[ins.display_section == "audience_intelligence"].sort_values(["field_order", "rank"])
    cols = st.columns(2)
    for i, (label, grp) in enumerate(ai.groupby("display_label", sort=False)):
        with cols[i % 2]:
            st.markdown(f"**{label}**")
            for r in grp.itertuples():
                st.markdown(f"- {r.statement} — {pct(r.audience_support_rate, 1)} of mapped audience signals")

    st.markdown("#### Audience concept profile")
    ap = prof[prof.layer == "audience"].sort_values("comment_prevalence", ascending=False).head(12)
    if not ap.empty:
        chart = ap.assign(share=ap.comment_prevalence * 100).set_index("canonical_concept")[["share"]]
        st.bar_chart(chart, horizontal=True, x_label="% of annotated comments", y_label="")
        st.dataframe(
            ap[["canonical_category", "canonical_concept", "unique_comment_count", "comment_prevalence"]]
              .rename(columns={"canonical_category": "category", "canonical_concept": "concept",
                               "unique_comment_count": "comments", "comment_prevalence": "share"}),
            hide_index=True, width="stretch",
            column_config={"share": st.column_config.NumberColumn(format="%.3f")},
        )

    st.markdown("#### Most-liked comments behind audience concepts")
    top = (ev[(ev.layer == "audience")].dropna(subset=["evidence_likes"])
             .drop_duplicates("evidence_text").sort_values("evidence_likes", ascending=False).head(6))
    for r in top.itertuples():
        st.markdown(f"> {r.evidence_text}")
        st.caption(f"→ {r.canonical_concept} · 👍 {int(r.evidence_likes):,}")

# ---------------------------------------------------------------------------
# Evaluation & System
# ---------------------------------------------------------------------------

with tab_system:
    st.markdown("#### Evaluation at a glance")
    funnel = metrics_section("pipeline_funnel")
    sig = metrics_section("eval_signal_detection")

    metric_values = dict(zip(funnel.metric_key, funnel.value))
    annotation_total = (
        float(metric_values.get("annotations_3a", 0))
        + float(metric_values.get("annotations_3b", 0))
    )
    hero_cards = [
        ("Videos analyzed", funnel.loc[funnel.metric_key.eq("videos"), "display_value"].iloc[0]),
        ("Viewer comments", funnel.loc[funnel.metric_key.eq("comments"), "display_value"].iloc[0]),
        ("Taxonomy annotations", f"{int(annotation_total):,}"),
        ("Human-QC F1", f"{float(sig.loc[sig.metric_key.eq('signal_f1'), 'value'].iloc[0]):.1%}"),
        ("Governed ontology nodes", funnel.loc[funnel.metric_key.eq("ontology_v02_nodes"), "display_value"].iloc[0]),
        ("Approved relationship rules", funnel.loc[funnel.metric_key.eq("relationship_rules_approved"), "display_value"].iloc[0]),
    ]
    cols = st.columns(3)
    for i, (label, value) in enumerate(hero_cards):
        cols[i % 3].metric(label, value)

    with st.expander("View full pipeline metrics"):
        cols = st.columns(4)
        for i, r in enumerate(funnel.itertuples()):
            cols[i % 4].metric(
                r.label,
                r.display_value,
                help=r.note if isinstance(r.note, str) and r.note else None,
            )

    st.markdown("#### Audience extraction vs. human review")
    qual = metrics_section("eval_interpretation_quality")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Signal detection**")
        top4 = sig[sig.metric_key.isin(["signal_precision", "signal_recall", "signal_f1", "signal_accuracy"])]
        mc = st.columns(4)
        for col, r in zip(mc, top4.itertuples()):
            col.metric(r.label, f"{float(r.value):.1%}")
        v = dict(zip(sig.metric_key, sig.value.astype(float)))
        cm = pd.DataFrame(
            {"Human: signal": [v.get("true_positive"), v.get("false_negative")],
             "Human: no signal": [v.get("false_positive"), v.get("true_negative")]},
            index=["Model: signal", "Model: no signal"],
        ).astype("Int64")
        st.dataframe(cm, width="stretch")
        n = sig[sig.metric_key == "qc_sample_size"].display_value
        st.caption(f"Human-QC sample n = {n.iloc[0] if len(n) else '—'} comments.")
    with c2:
        st.markdown("**Interpretation quality**")
        mc = st.columns(4)
        for col, r in zip(mc, qual.itertuples()):
            col.metric(r.label, r.display_value)
        st.markdown("**Error analysis**")
        for r in metrics_section("eval_error_analysis").itertuples():
            st.markdown(f"- {r.label}: {r.display_value} ({r.note})")

    st.markdown("**What human evaluation changed in the extraction / ontology policy**")
    for r in metrics_section("eval_findings").itertuples():
        st.markdown(f"- **{r.label}** — {r.note}")

    st.markdown("#### Governed relationship rules (v0.1)")
    st.caption("Cross-ontology links are proposed by the system, then approved, rejected, or deferred by a human reviewer.")
    rules = D["rules"]
    only_demo = st.toggle("Only rules touching this demo", value=True)
    rshow = rules[rules.used_in_demo] if only_demo else rules
    st.dataframe(
        rshow[["source_layer", "source_concept", "relationship_type", "target_layer", "target_concept",
               "relationship_strength", "rule_status"]],
        hide_index=True, width="stretch",
    )
    st.caption(" · ".join(f"`{k}`: {v}" for k, v in REL_EXPLAIN.items()))

    st.markdown("#### Scoring configuration")
    for r in metrics_section("scoring_config").itertuples():
        st.markdown(f"- {r.label}: **{r.display_value}**")
    if isinstance(case.interpretation_scope, str):
        st.warning(f"**Interpretation scope.** {case.interpretation_scope}")

    st.markdown("#### Scope & limitations")
    st.markdown(
        "- 9 videos from one artist's ecosystem; comments are the platform's relevance-sorted sample.\n"
        "- Inferred framing is a content-framing proxy derived from observable signals; it is not the creator's self-reported internal intent.\n"
        "- Not every annotation maps into the governed ontology. Unmapped labels are kept but not scored.\n"
        "- *No governed counterpart* ≠ *viewers didn't perceive it*. It means the ontology and rules don't support the claim yet.\n"
        "- Annotations are LLM-generated. Audience extraction is validated on a human-reviewed sample."
    )

    man = D["manifest"]
    with st.expander("Data provenance"):
        st.caption(f"Snapshot generated {man.get('generated_at', '—')} · versions: "
                   + ", ".join(f"{k} {v}" for k, v in man.get("versions", {}).items()))
        st.json(man.get("sources", {}), expanded=False)

st.divider()
st.caption(
    (f"Built by {AUTHOR} · " if AUTHOR else "")
    + "Pipeline: YouTube ingestion → LLM taxonomy extraction → ontology normalization with "
    "human review → governed relationship mapping → alignment diagnostics."
    + (f" · [Code on GitHub]({REPO_URL})" if REPO_URL else "")
)
