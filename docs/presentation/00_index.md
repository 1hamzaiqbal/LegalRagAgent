# Presentation Index

Read these in order:

1. `docs/presentation/00_index.md` - start here.
2. `docs/presentation/01_results_tables.md` - the numbers.
3. `docs/presentation/02_methods_explained.md` - what each method does.
4. `docs/presentation/03_takeaways.md` - the punchy bullet points.
5. `docs/presentation/04_datasets_and_models.md` - what we tested on.
6. `docs/presentation/05_logs_index.md` - where to find every cited detail log.
7. `docs/presentation/06_next_steps.md` - what is still open.

The citation gate is `docs/signoff_log.md`. If a number is not signed there, keep it out of the presentation or mark it explicitly as pending, partial, or direction-only.

## If someone asks me X, I look at...

| Question | Start with | Then verify in |
|---|---|---|
| What are the headline numbers? | `01_results_tables.md` | `docs/signoff_log.md` Sections A and B |
| What does `multi_hyde_diverse` actually do? | `02_methods_explained.md` | `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl` |
| Why is BarExam different from MuSiQue? | `03_takeaways.md` | `04_datasets_and_models.md` |
| What is the BarExam full-corpus evidence? | `01_results_tables.md` Table A/B | `05_logs_index.md` BarExam Tier 3 |
| What is the MuSiQue paper headline? | `01_results_tables.md` Table D/E | `05_logs_index.md` MuSiQue Llama 70B |
| Is the MuSiQue lift universal across models? | `03_takeaways.md` | `01_results_tables.md` Table G |
| Which logs can I cite directly? | `05_logs_index.md` | `docs/signoff_log.md` |
| Which runs are pending or unsafe? | `05_logs_index.md` in-flight and do-not-cite sections | `docs/signoff_log.md` Sections D and G |
| How do I pull a single record for a slide? | `05_logs_index.md` Quick how-to | The matching local detail log |

## Quick Q&A

- "What's our paper headline?" -> `03_takeaways.md`
- "Show me the numbers" -> `01_results_tables.md`
- "What does mhd actually do?" -> `02_methods_explained.md`
- "Where did this number come from?" -> `05_logs_index.md` -> detail log path
- "What's left to do?" -> `06_next_steps.md`
- "Why did we test on these models?" -> `04_datasets_and_models.md`
