# Datasets And Models

## BarExam

BarExam is the legal multiple-choice evaluation set in this repo. The source dataset is Hugging Face [`reglab/barexam_qa`](https://huggingface.co/datasets/reglab/barexam_qa), downloaded into `datasets/barexam_qa/`. The meeting-facing full-corpus tier is N=1195 questions, as signed in `docs/signoff_log.md` Section A. Questions combine a fact pattern, a legal stem, four answer choices, and a gold A-D option. A real local example is `mbe_0`: prompt: "Paul, the Plaintiff in a personal injury action, called Wes as a witness to testify that Dan's car, in which Paul had been riding. ran a red light. Wes, however, testified that Dan's car did not run the light." Question: "Paul then called Vic to testify that Dan's car did run the light. The trial judge should rule that Vic's testimony is"; choices A-D are admissibility/impeachment alternatives; gold answer is `B`. It is interesting because retrieval must find doctrine, but final scoring is exact legal-MC selection.

## MuSiQue

MuSiQue is the multi-hop open-domain QA dataset used for compositional reasoning checks. The source loader is Hugging Face [`dgslibisey/MuSiQue`](https://huggingface.co/datasets/dgslibisey/MuSiQue), and this repo prepares it into `datasets/musique/questions.csv` plus `datasets/musique/passages.csv`. The signoff gate treats MuSiQue full validation as N=2400. A real local example is `2hop__460946_294723`: question: "Who is the spouse of the Green performer?"; gold answer: `Miquette Giraudy`; supporting passages connect `Green` to Steve Hillage and Hillage to Miquette Giraudy. Unlike BarExam, the answer is a short text span rather than A-D, and the model must bridge entities across multiple paragraphs. That makes it a useful stress test for whether retrieval variants preserve competing candidate paths or prematurely lock onto the wrong hop.

## Gemma 4 26B-A4B

Gemma 4 26B-A4B is the cluster headline mixture-of-experts model. Repo model notes record it as `google/gemma-4-26B-A4B-it`, roughly 25B total parameters with about 3.8-4B active per token. The signed BarExam results use `cluster-vllm`, served on the WUSTL cluster via the Gemma 4 vLLM environment. Some MuSiQue attempts used OpenRouter as `or-gemma4-26b`, but `docs/signoff_log.md` Section D' documents runaway-loop generations there: occasional 600s-2400s calls, repetitive answers, and a killed full run. For citations, prefer cluster vLLM Gemma results when available.

## Gemma 4 E4B

Gemma 4 E4B is the smaller Gemma 4 model in the deck. Repo throughput notes name it `google/gemma-4-E4B-it` and record it as about 8B / 4.5B effective in this setup. It is served through the same cluster vLLM path as the larger Gemma 4 model, with cached weights under the cluster Hugging Face cache. Its role is the small-model confirmation point for BarExam: the signoff log shows the same `rag_snap_hyde` winner pattern at N=1195. The main caveat is coverage, not provider behavior: the presentation cites the signed BarExam matrix, while MuSiQue E4B is not a landed signed result here.

## Llama 3.3 70B Dense

Llama 3.3 70B is the dense paper-headline model for MuSiQue. The repo records it as a 70B dense model, served primarily by Groq as `groq-llama70b`. It anchors the N=200 MuSiQue matrix in `docs/signoff_log.md` Section B.1: all eight method rows have local detail logs and paired McNemar support. Serving caveats are ordinary provider limits rather than a signed correctness confound: the repo notes Groq request/token rate limits, so parallel Llama jobs can contend, but the cited N=200 logs landed cleanly enough for the signoff gate.

## Qwen3 30B MoE

Qwen3 30B MoE is the third model class: a non-Gemma mixture-of-experts checkpoint, recorded in `CLAUDE.md` as 30B total with about 3B active. The cited local MuSiQue rows use OpenRouter as `or-qwen3-30b-moe`; the BarExam cross-family board also uses Qwen-family API rows, but those are support-only, not the paper headline. Its signed MuSiQue status is still directional: N=100 `rag_simple` and `multi_hyde_diverse` are approved with caveats, and the N=2400 `qwen_full` run is marked in flight in `docs/signoff_log.md` Section D with no local source log yet. Cite it as a model-class check, not a completed Tier 3 claim.
