# Demo Decisions Log

Terminology and framing decisions for the recruiter-facing demo, and the reasoning behind each.
New entries go at the top.

---

## 2026-10-02 · P0-D deploy prep

- README rewritten under the product name **Content Alignment Evaluation Platform**, using Framing
  terminology, "diagnostics, not verdicts", and the inferred-framing limitation.
- **Percent format:** signal-detection metrics show as one-decimal percentages everywhere (UI, data, README):
  precision 90.0%, recall 100.0%, F1 94.7%, accuracy 92.0%.
- **Evidence language:** Alignment Journey keeps evidence in its original language. Caption above Evidence:
  "Evidence is shown in its original language where available; English summaries are provided in the
  Overview for the featured path."
- `REPO_URL` set to `github.com/dolcefarnienteleone/content-alignment-evaluation-platform`. `AUTHOR` is filled
  in by the repo owner.

---

## 2026-10-02 · P0-C hero polish

- **Path diagram first.** Overview now opens with "The path through the engine": Inferred framing →
  Content understanding → Audience perception, with `relationship_type · strength` on each link. It is
  built from focus pairs in `demo_alignment_pairs.csv`, so every case gets it. A missing audience link
  renders as a dashed "No governed audience counterpart" node with a "no approved rule" edge (gap case).
  On mobile it stacks vertically, with layer labels inside each node.
- **Evidence at a glance.** Under the path: "What the content shows" (one framing quote and one content
  quote, de-duplicated) next to "What viewers said" (top-liked comments for the linked audience concept).
  When there is no governed link, the right column becomes **"What viewers expressed instead"**, with a note
  that these are not counted as alignment. Model explanations are shown only when they are in English.
- **Scores read as facts.** Each diagnostic now has a sentence computed from the pair table, e.g.
  "**1 of 5** content concepts have a governed audience counterpart. The strongest, *Vocal Ability*,
  appears in 23% of mapped audience signals." This explains why 0.08 is low without implying it's a failure.
- **Focus evidence expanded by default** in Alignment Journey.
- **Terminology sync in data:** `build_demo_dataset.py` insight labels and `comparison_label` now say
  "Framing …" instead of "Intent …" (e.g., "Framing realized in content", "Framing → Audience").

---

## 2026-10-02 · P0-B review pass (semantics + evaluation hierarchy)

### Terminology: "Inferred framing", not "creator intent"

| Internal (data / pipeline) | Shown in the UI | Why |
|---|---|---|
| `inferred_content_intent` / layer `intent` | **Inferred framing** | It is inferred from observable signals (title, description, opening/closing segments). It is not the creator's self-reported internal intent. |
| `ai_content_understanding` / layer `content` | **Content understanding** | What the content itself shows. |
| `audience_perception` / layer `audience` | **Audience perception** | Comment-level perception. |
| `intent_to_content` | **Framing → Content** | "Was the inferred framing realized in the content?" |
| `content_to_audience` | **Content → Audience** | "Which content signals carried through to audience perception?" |
| `intent_to_audience` | **Framing → Audience** | "Which inferred framing signals carried through end-to-end?" |

Column names in `data/demo/*.csv` keep the pipeline's `intent_*` names. Only the UI labels change, so the
demo data stays traceable to pipeline outputs.

Scope & Limitations states this explicitly:
> Inferred framing is a content-framing proxy derived from observable signals; it is not the creator's
> self-reported internal intent.

### "Source concept", not "creator-side concept"

- In Content → Audience, the source is content understanding, not creator framing. Generic copy says
  **source concept → target side**.
- `REL_EXPLAIN` no longer implies every source is creator-side:
  `contributes_to` = "source concept contributes to the target-side perception";
  `manifests_as` = "source concept shows up as a different target-side perception".
- The unmapped-label expander is now **"Framing/content labels not in the governed ontology"**.

### Scores are diagnostics

- Section renamed **"Alignment V1 scores" → "Alignment diagnostics"**.
- Copy: scores combine governed concept coverage, relationship strength, confidence, and audience support.
  They are **not** content-quality ratings, model-accuracy metrics, or literal percentages of audience alignment.
- There is still no blended overall score. The three comparisons stay separate.

### Findings wording: conservative, not prescriptive

- Overview finding columns: **✅ What carried through · ◌ What remained unmatched · + What audiences added**.
- In `build_demo_dataset.py`, `unexpected_perception` and `emergent_audience_signal` changed polarity from
  `opportunity` to `signal`. Emergent audience signals are kept separate and are **not** automatically
  treated as product opportunities.
- Case stories were rewritten to match:
  - **Hero:** "The inferred content framing emphasizes vocal production craftsmanship…"
  - **Gap:** "Caring is clear in the governed content profile, but the system does not overclaim audience
    perception." The gap is preserved as uncertainty and is not turned into a claim about viewers.
  - **Relationship:** "Relationship identity carries through in multiple forms." Chemistry, cuteness, and love
    are kept as emergent audience signals, not opportunities.

### Product name and header

- App title: **Content Alignment Evaluation Platform**.
- Header question: "How does creator framing carry through content and audience perception?"
- Footer architecture: ingestion → LLM taxonomy extraction → ontology normalization with human review →
  **governed relationship mapping → alignment diagnostics**.

### Evaluation tab: recruiter-first hierarchy

Six headline cards first:
**Videos analyzed · Viewer comments · Taxonomy annotations (988 = 3A + 3B) · Human-QC F1 · Governed ontology
nodes (123) · Approved relationship rules**.
The full pipeline funnel is in a "View full pipeline metrics" expander. The section heading is
"What human evaluation changed in the extraction / ontology policy".

### Open follow-ups

- [x] `README.md` still uses the old name ("Creator ↔ Audience Alignment Engine") and "Intent → Content".
      Sync it with the UI before P0-D. *(Done in P0-D.)*
- [x] Some `model_explanation` values for content-layer evidence are in Korean (relationship case). Either
      ask for English explanations in a future 3A run or label them as model output. *(P0-D: evidence stays in its original language, with a caption above Evidence. The Overview shows English summaries for the featured path.)*
- [x] Human-QC F1 shows as `94.7%` in the headline card and `0.95` in the signal-detection block. Pick one format. *(P0-D: one-decimal percentages everywhere, e.g. 94.7%.)*

---

## 2026-10-01 · P0-A / P0-B foundations

- The demo reads only `data/demo/` (static snapshot): no API key, no live LLM, no local paths.
- Three curated cases, locked: hero `w0wF-O0EGsQ`, gap `gaw2OJ5PoDc`, relationship `v5DXpxUAO3w`.
- `build_demo_dataset.py` checks that each curated story still holds in the data and fails the build if one doesn't.
- Viewer comments keep text and like count only. Author IDs and comment IDs are excluded.
- Cases are deep-linkable: `?case=hero | gap | relationship`.
