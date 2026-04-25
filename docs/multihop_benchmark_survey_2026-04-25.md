# Multi-hop QA benchmark survey — 2026-04-25

Survey to pick the first multi-hop RAG benchmark to add to our harness
alongside BarExam. Three candidates evaluated: HotpotQA, MuSiQue,
2WikiMultihopQA.

## Spec cards

### 1. HotpotQA
- **HF**: `hotpotqa/hotpot_qa`
- **Splits**: distractor train 90,447 / val 7,405; fullwiki train 90,447 / val 7,405 / test 7,405
- **Configs**: `distractor` (gold + 8 distractor paragraphs/Q) and `fullwiki` (no context — full Wikipedia retrieval)
- **Question format**: open-ended natural language; sub-types `bridge` (multi-hop chain) and `comparison`
- **Answer format**: short span 1–~30 tokens; includes yes/no for comparisons
- **Context format**: `context.title: List[str]`, `context.sentences: List[List[str]]` (paragraphs as sentence lists)
- **Gold-passage labels**: yes — sentence-level via `supporting_facts.{title, sent_id}`
- **Distractors**: yes — distractor config bundles ~10 paragraphs (2 gold + 8 distractor)
- **Metric**: EM + F1 (joint answer + supporting-facts F1)
- **Wiring complexity**: Low–Medium. Reuse Legal-RAG-QA formatter; need EM/F1 scorer; sentence-level gold means `golden_passage` needs to concat sentences per title

### 2. MuSiQue
- **HF**: `dgslibisey/MuSiQue` (community mirror of MuSiQue-Ans)
- **Splits**: train 19.9k / val 2.42k (test labels withheld)
- **Configs**: single `default`. Upstream has `MuSiQue-Ans` (answerable) + `MuSiQue-Full` (with unanswerable)
- **Question format**: open-ended, 2–4 hop compositional Qs explicitly built from single-hop chains
- **Answer format**: short span/entity 1–~10 tokens; includes `answer_aliases` for flexible matching
- **Context format**: `paragraphs: [{idx, title, paragraph_text}]` — flat list, ~20 paragraphs/Q
- **Gold-passage labels**: yes — `question_decomposition[].paragraph_support_idx` per hop (ordered chain)
- **Distractors**: yes — curated hard distractors specifically designed to defeat shortcut models
- **Metric**: EM + F1 (with alias matching)
- **Wiring complexity**: Low. Cleanest schema; flat paragraph list; explicit decomposition field doubles as a benchmark for `decompose_rag` quality

### 3. 2WikiMultihopQA
- **HF canonical**: `voidful/2WikiMultihopQA`. GPT-paraphrased fork: `scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35`
- **Splits**: canonical train ~167k / val ~12.6k / test ~12.6k
- **Configs**: `default` only
- **Question format**: open-ended; types `comparison`, `inference`, `compositional`, `bridge_comparison`
- **Answer format**: short span 2–~70 chars
- **Context format**: `context.{title, content}` — HotpotQA-style sentence arrays
- **Gold-passage labels**: yes — `supporting_facts.{title, sent_id}` PLUS structured `evidences` Wikidata triples (only one with KB ground truth)
- **Distractors**: yes
- **Metric**: EM + F1 (joint + supporting-facts F1)
- **Wiring complexity**: Low–Medium. Near-identical to HotpotQA so any HotpotQA loader transfers. Canonical mirror has dataset-viewer timeout — may need parquet download

## Recommendation: lead with MuSiQue

**Wiring**: flat paragraph list (no nested sentence arrays), explicit
`paragraph_support_idx` makes `golden_passage` a one-liner, `answer_aliases`
plugs into a tolerant string-match scorer.

**Narrative**: hardest of the three; published baselines show 30–50pp gaps
between single-hop methods and multi-hop methods (IRCoT, Self-Ask paper) —
exactly the regime where snap+HyDE should shine. HotpotQA is partially
solvable by single-hop shortcuts (the critique that motivated MuSiQue),
so a snap+HyDE win there is less convincing. 2WikiMultihopQA's
Wikidata-template construction feels synthetic vs MuSiQue's compositional
naturalness.

**Bonus**: `question_decomposition` gives free ground-truth for the
`decompose_rag` mode → built-in diagnostic for *why* the method works.

## Next: HotpotQA as second benchmark for breadth

After MuSiQue lands, add HotpotQA as the secondary benchmark since
"everyone runs it, so reviewers expect it". 2WikiMultihopQA only if a
third dataset is needed for robustness claims.

## Implementation notes for the MuSiQue loader

1. Add to `eval/eval_config.py`:
   - `format_musique_prompt(row)` — open-ended formatter, similar to `format_open_prompt`
   - `extract_answer_musique(text)` — alias-tolerant string matcher
2. Add to `eval/eval_harness.py`:
   - Dataset loading branch in main eval loop
   - `_retrieval_question` already prompt-aware — verify it handles MuSiQue
   - Wire `gold_passage` as `paragraphs[support_idx[0]].paragraph_text` (or concat all hop golds)
3. Add MuSiQue paragraphs to ChromaDB collection `musique_passages`
   - Or skip retrieval embedding for first smoke and use the in-question paragraphs directly
4. Smoke test: `--mode rag_simple --dataset musique --questions 50` on E4B
