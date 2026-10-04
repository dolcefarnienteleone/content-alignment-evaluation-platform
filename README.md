# Content Alignment Evaluation Platform

**How does creator framing carry through content and audience perception?**

This system analyzes YouTube content and its comment sections. It compares three layers:

- **Inferred framing**: what the content is framed to be about, inferred from observable signals
- **Content understanding**: what the content actually shows
- **Audience perception**: what viewers took away, from their comments

All three layers are mapped into a **human-governed ontology** and linked through **human-approved
relationship rules**. For every video, the system shows with evidence what carries through, what remains
unmatched, and what else audiences bring up. It does not reduce this to a single score.

**▶ Live demo:** _[content-alignment-eval.streamlit.app](https://content-alignment-eval.streamlit.app/?case=hero)_ · no login, no API key, no live LLM

---

## Three curated cases

| Case | Path through the engine | What it shows |
|---|---|---|
| **Creative Craft** (hero)<br>[`w0wF-O0EGsQ`](https://www.youtube.com/watch?v=w0wF-O0EGsQ) | Vocal Production Craftsmanship → Vocal Production Craftsmanship → **Vocal Ability** | Cross-ontology alignment. Creator-side and audience-side concepts don't need identical labels to be linked by a governed `contributes_to` relationship. |
| **Values / Care** (gap)<br>[`gaw2OJ5PoDc`](https://www.youtube.com/watch?v=gaw2OJ5PoDc) | Caring → Caring → *(no governed audience counterpart)* | Evaluation discipline. A missing governed counterpart is kept as uncertainty. It is not turned into a claim about audience perception. |
| **Relationship Identity**<br>[`v5DXpxUAO3w`](https://www.youtube.com/watch?v=v5DXpxUAO3w) | Playful Camaraderie → **Playful Teasing Dynamic** · Friendship → Friendship | Supports both direct and cross-ontology carry-through, while keeping emergent audience signals analytically separate. |

## How it works

```
YouTube video ─► 3A  Content extraction ─► inferred framing + content understanding ─┐
                                                                                     ├─► free-form labels (894)
Comments ──────► 3B  Audience extraction ─► audience perception ─────────────────────┘           │
                  └─ 3B.1 human QC: precision 90.0% · recall 100.0% · F1 94.7%                    ▼
                                         4  Ontology normalization (894 → 658 groups → v0.2: 123 nodes)
                                            LLM proposals + human review ⏸                       │
                                                                                                  ▼
                                         5  Alignment diagnostics
                                            5A normalized concept profiles
                                            5B governed relationship mapping (human-approved rules ⏸)
                                            5C diagnostics: Framing→Content · Content→Audience · Framing→Audience
                                            5D interpretable insights + evidence
```

### Key design choices

- **No exact-label matching across ontologies.** Creators and audiences describe the same thing differently
  ("vocal production craftsmanship" vs. "vocal ability"). Cross-side links come only from an exact canonical
  match or a reviewed rule set (`relationship_rules_v0.1`: 32 approved, 7 rejected, 1 deferred), with typed
  relations such as `contributes_to` and `manifests_as`.
- **Humans govern the vocabulary.** LLMs propose and people approve, at three checkpoints: audience-extraction QC
  (3B.1), ontology review (4C), and relationship rules (5B.1).
- **Diagnostics, not verdicts.** Scores combine governed concept coverage, relationship strength, confidence,
  and audience support. They are not content-quality ratings, model-accuracy metrics, or literal percentages of
  audience alignment. The three comparisons stay separate, with no blended overall score.
- **Abstain rather than stretch.** When no governed relationship exists, the system reports the gap and does not
  count a weak semantic similarity as a match. *No governed counterpart* ≠ *viewers didn't perceive it*.
- **Emergent audience signals stay separate.** Perceptions the framing didn't anticipate are surfaced as
  audience signals. They are not automatically treated as product opportunities.

## Evaluation

Audience extraction (3B) compared with human review on a stratified sample of n = 50 comments:

| Signal detection | | Interpretation quality | |
|---|---|---|---|
| Precision | 90.0% | Overall correct | 86% |
| Recall | 100.0% | Correct + partial | 92% |
| F1 | 94.7% | Category validity | 80% |
| Accuracy | 92.0% | Label validity | 90% |

Errors: over-interpretation 10%, under-interpretation 6%. What human evaluation changed in the extraction and
ontology policy:

- Generic interaction ≠ community signal
- Emoji-only ≠ sufficient signal
- Content callbacks → community-signal candidates
- Member and fandom references can be under-detected

## Repository layout

```
app.py                          Streamlit demo (reads data/demo/ only)
data/demo/                      static snapshot for the 3 curated cases
scripts/build_demo_dataset.py   pipeline outputs → data/demo/ (checks each case's story against the data)
pipeline/
  stage1_ingestion/             YouTube metadata, transcripts, comments
  stage3_extraction/            3A / 3B LLM extraction, 3B.1 human-QC evaluation
  stage4_ontology/              4A–4D taxonomy normalization → ontology v0.2
  stage5_alignment/             5A–5D governed relationship mapping + alignment diagnostics
  config/selected_videos.csv    input video list
run_pipeline.sh                 runs every stage in order, stopping at human checkpoints
docs/PIPELINE.md                stage-by-stage inputs / outputs
```

## Run it

**Demo (no keys needed)**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Rebuild the demo snapshot** from a workspace that has the pipeline outputs under `data/`:
```bash
python scripts/build_demo_dataset.py --source-root /path/to/workspace
```
The build fails if a curated story no longer holds in the data. For example, the hero case must still carry
`Vocal Production Craftsmanship -> Vocal Ability` through to the audience.

**Full pipeline** (needs YouTube and OpenAI keys; see `.env.example`):
```bash
pip install -r requirements-pipeline.txt
./run_pipeline.sh ingest
./run_pipeline.sh extract
# ./run_pipeline.sh lists every stage and human checkpoint
```
Intermediate pipeline data is not committed. Only the curated snapshot in `data/demo/` is.

## Scope & limitations

- 9 videos from one artist's ecosystem (TXT). Comments are the platform's relevance-sorted sample, not full threads.
- Inferred framing is a content-framing proxy derived from observable signals. It is not the creator's
  self-reported internal intent.
- 68% of annotations map into the governed ontology v0.2. Unmapped labels are kept but not scored.
- Annotations are LLM-generated. Audience extraction is validated on a human-reviewed sample.
- Evidence is shown in its original language (mostly Korean and English). English summaries are provided for each
  case's featured path.
