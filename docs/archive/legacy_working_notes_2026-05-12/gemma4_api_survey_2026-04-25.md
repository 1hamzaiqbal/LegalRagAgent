# Gemma 4 API provider survey — 2026-04-25

Question: can we parallelize Gemma 4 evals via API instead of cluster vLLM?

## Provider availability matrix

| Provider | E4B-it | 26B-A4B-it | 31B-it | Status | Free tier | Paid pricing |
|---|---|---|---|---|---|---|
| Google AI Studio (Gemini API) | ❌ | ✅ | ✅ | Live | **Yes — Gemma free; ~30 RPM, 14.4K RPD** | n/a (Gemma is free) |
| OpenRouter (paid) | ❌ | ✅ | ✅ | Live, 10 routed providers | n/a | $0.06/M in, $0.33/M out, 256K ctx |
| OpenRouter (`:free`) | ❌ | ✅ | ✅ | Live | 20 RPM, ~200 RPD (failed calls count) | n/a |
| Together AI | ❌ | ❌ | ✅ | Live serverless | None | $0.20/$0.50 per M |
| Fireworks AI | ❌ | dedicated only | dedicated only | Not serverless | None | Rent GPU |
| Cerebras | ❌ | ❌ | ❌ | Not hosted | n/a | n/a |
| Groq | ❌ | ❌ | ❌ | Not hosted (still on Gemma 2 9B) | n/a | n/a |
| Cloudflare Workers AI | ❌ | ✅ | ❌ | Live | Workers free tier | per-token |

## Key findings

1. **E4B-it has NO commercial API** — Google positions it as edge/on-device only (Edge Gallery, Ollama, llama.cpp). For E4B we MUST use cluster vLLM.
2. **26B-A4B-it is FREE on Google AI Studio** — Gemma served free at Tier 1 limits. ~14.4K requests/day fits ~1,200 questions × 8 calls = 9,600 calls per mode comfortably.
3. **31B-it has more options** — Google AI Studio (free), Together ($0.20/$0.50), OpenRouter (paid).

## Cost estimate per N=1195 mode (paid OpenRouter)

| Mode | Calls/q | Total calls | Input tokens (~M) | Output tokens (~M) | Est cost |
|---|---|---|---|---|---|
| rag_simple | 1 | 1,195 | 0.5 | 0.5 | $0.20 |
| rag_hyde | 2 | 2,390 | 1.0 | 1.0 | $0.40 |
| rag_snap_hyde | 3 | 3,585 | 1.5 | 1.5 | $0.60 |
| subagent_rag | 4 | 4,780 | 2.0 | 2.0 | $0.80 |

Total full 10-mode wave (26B): ~$5–8 paid, or **$0 free on Google AI Studio**.

## Action items

1. **Add `GOOGLE_API_KEY` to `.env`** to unlock free 26B-A4B Gemma 4 path
2. **Add Gemma 4 model entries** to `llm_config.py` (existing pattern for `gemma-3-*`):
   ```python
   "gemma4-26b-a4b":  ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemma-4-26b-a4b-it", 14_400, None),
   "gemma4-31b":      ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemma-4-31b-it",     14_400, None),
   "or-gemma4-26b":   ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-4-26b-a4b-it",   None, None),
   ```
3. **Run a smoke test** (`--mode rag_simple --provider gemma4-26b-a4b --questions 10`) to verify model parity with cluster vLLM
4. **If parity holds**, fire seed=99 variance runs OR MuSiQue smoke via API while cluster handles E4B + the seed=42 26B wave

## Recommendation

Once GOOGLE_API_KEY is set and parity smoke-test passes, **use the API path for 26B variance work and MuSiQue smoke** — keeps cluster GPUs free for E4B (which has no API alternative) and adds parallelism for free.

E4B stays on cluster vLLM (no choice).

## Sources

- ai.google.dev/gemma/docs/core/model_card_4
- ai.google.dev/gemma/docs/core/gemma_on_gemini_api
- openrouter.ai/google/gemma-4-26b-a4b-it (paid + `:free`)
- openrouter.zendesk.com (rate limits)
- together.ai/models/gemma-4-31b
- fireworks.ai/models/fireworks/gemma-4-26b-a4b-it
- inference-docs.cerebras.ai/models/overview
- console.groq.com/docs/models
- developers.cloudflare.com/changelog/post/2026-04-04-gemma-4-26b-a4b-workers-ai/
