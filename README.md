# LegalRagAgent

Agentic Legal RAG system built on LangGraph. Uses a **classify-plan-execute-evaluate** loop to answer legal research questions by retrieving passages from a ChromaDB vector store of bar exam materials.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your LLM

Copy the example env file and add your API key:

```bash
cp .env.example .env
```

The default configuration uses **Groq free tier** (no credit card needed). Get an API key at [console.groq.com](https://console.groq.com) and paste it in `.env`.

Other supported backends: Ollama (local), OpenAI — see `.env.example` for config.

### 3. Load passages (first run only)

If the ChromaDB store hasn't been populated yet:

```bash
python rag_utils.py
```

This loads the first 1000 bar exam passages into `./chroma_db/`.

### 4. Run the agent

```bash
python main.py simple       # "What are the elements of a negligence claim?"
python main.py multi_hop    # Constitutional rights + 4th/5th Amendment search scenario
python main.py medium       # Preliminary injunction standard and factors
```

## Architecture

Four-node LangGraph state machine:

```
classifier → planner → executor → evaluator → {executor | END}
```

- **Classifier**: Routes queries as `simple` (1 step) or `multi_hop` (2-4 steps)
- **Planner**: LLM generates a structured research plan
- **Executor**: Per step — rewrites query, retrieves from ChromaDB, synthesizes answer, grounds with citations
- **Evaluator**: Checks confidence (cosine similarity). Passes at >= 0.7, injects sub-step on failure

### Skills (5 prompt files)

| Skill | Purpose |
|-------|---------|
| `classify_and_route.md` | Classify query complexity |
| `plan_synthesis.md` | Generate research plan |
| `query_rewrite.md` | Optimize retrieval queries |
| `synthesize_answer.md` | Synthesize grounded answers |
| `ground_and_cite.md` | Verify grounding + add citations |

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams and the 12-skill vision.

## Retrieval Evaluation

```bash
python eval.py
```

Measures Recall@5 and MRR on 200 bar exam QA pairs.

## Required Skills

The LangGraph agent dynamically loads agent instructions (skills) from markdown files located in the `skills/` directory. All 5 skill files are included in this repo.
