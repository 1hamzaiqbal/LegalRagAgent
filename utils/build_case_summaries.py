"""Build case-level summaries from the barexam corpus.

Groups passages by case_id, concatenates them, and generates a 2-3 sentence
summary per case using an LLM. Output: CSV with case_id, n_paragraphs,
topic_keywords, summary.

Usage:
  # Via vLLM on cluster:
  python utils/build_case_summaries.py --base-url http://127.0.0.1:8010/v1 --model google/gemma-4-E4B-it

  # Via API:
  python utils/build_case_summaries.py --provider deepseek

  # Resume interrupted build:
  python utils/build_case_summaries.py --resume --base-url http://127.0.0.1:8010/v1 --model google/gemma-4-E4B-it
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPUS_CSV = "datasets/barexam_qa/barexam_qa_train.csv"
OUTPUT_CSV = "datasets/barexam_qa/case_summaries.csv"
OUTPUT_JSONL = "datasets/barexam_qa/case_summaries.jsonl"

SUMMARY_SYSTEM = (
    "You are a legal research assistant. Given excerpts from a court case or legal reference, "
    "write a 2-3 sentence summary covering: (1) the core legal issue or doctrine, "
    "(2) the key holding or rule, (3) the area of law (e.g., torts, contracts, criminal law). "
    "Be specific and use proper legal terminology. No bullet points."
)

MAX_INPUT_CHARS = 3000  # first ~750 words of each case


def build_cases(df: pd.DataFrame) -> list[dict]:
    """Group passages by case_id into case documents."""
    cases = []

    # Handle MBE passages (no case_id) — each is its own "case"
    mbe = df[df['source'] == 'mbe'].copy()
    for _, row in mbe.iterrows():
        cases.append({
            'case_id': f"mbe_{row['idx']}",
            'source': 'mbe',
            'n_paragraphs': 1,
            'text': str(row['text'])[:MAX_INPUT_CHARS],
            'passage_ids': [str(row['idx'])],
        })

    # Handle wex passages — each is its own entry
    wex = df[df['source'] == 'wex'].copy()
    for _, row in wex.iterrows():
        cases.append({
            'case_id': f"wex_{row['idx']}",
            'source': 'wex',
            'n_paragraphs': 1,
            'text': str(row['text'])[:MAX_INPUT_CHARS],
            'passage_ids': [str(row['idx'])],
        })

    # Handle caselaw — group by case_id
    caselaw = df[df['source'] == 'caselaw'].dropna(subset=['case_id']).copy()
    caselaw['case_id'] = caselaw['case_id'].astype(int).astype(str)
    for case_id, group in caselaw.groupby('case_id'):
        group = group.sort_values('relative_paragraph_id')
        full_text = "\n\n".join(group['text'].astype(str).tolist())
        cases.append({
            'case_id': str(case_id),
            'source': 'caselaw',
            'n_paragraphs': len(group),
            'text': full_text[:MAX_INPUT_CHARS],
            'passage_ids': group['idx'].astype(str).tolist(),
        })

    return cases


def summarize_case(case: dict, llm_call_fn) -> str:
    """Generate summary for a single case."""
    user = f"## Legal Text ({case['source']}, {case['n_paragraphs']} paragraphs)\n\n{case['text']}"
    return llm_call_fn(SUMMARY_SYSTEM, user)


def make_llm_call_fn(args):
    """Create an LLM call function based on args."""
    if args.base_url:
        # vLLM / OpenAI-compatible API
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key or "DUMMY")
        model = args.model

        def call(system, user):
            # Gemma needs system+user merged
            if 'gemma' in model.lower():
                messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            else:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=200, temperature=0.0)
            return resp.choices[0].message.content.strip()
        return call
    else:
        # Use llm_config
        from llm_config import get_llm
        os.environ.setdefault("LLM_PROVIDER", args.provider)
        llm = get_llm(temperature=0.0)

        def call(system, user):
            from langchain_core.messages import SystemMessage, HumanMessage
            return llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip()
        return call


def main():
    parser = argparse.ArgumentParser(description="Build case summaries from barexam corpus")
    parser.add_argument("--base-url", type=str, help="vLLM base URL (e.g., http://127.0.0.1:8010/v1)")
    parser.add_argument("--model", type=str, default="google/gemma-4-E4B-it", help="Model name")
    parser.add_argument("--api-key", type=str, default="DUMMY", help="API key")
    parser.add_argument("--provider", type=str, default="deepseek", help="LLM provider (if not using base-url)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0=all)")
    parser.add_argument("--batch", type=str, default="all", help="Which batch: 'mbe', 'wex', 'caselaw', or 'all'")
    args = parser.parse_args()

    print(f"Reading corpus from {CORPUS_CSV}...")
    df = pd.read_csv(CORPUS_CSV)
    print(f"  {len(df):,} passages")

    print("Building cases...")
    cases = build_cases(df)
    print(f"  {len(cases):,} cases ({sum(1 for c in cases if c['source']=='caselaw')} caselaw, "
          f"{sum(1 for c in cases if c['source']=='mbe')} mbe, "
          f"{sum(1 for c in cases if c['source']=='wex')} wex)")

    if args.batch != "all":
        cases = [c for c in cases if c['source'] == args.batch]
        print(f"  Filtered to {args.batch}: {len(cases)} cases")

    if args.max_cases > 0:
        cases = cases[:args.max_cases]
        print(f"  Limited to {len(cases)} cases")

    # Load existing summaries for resume
    done_ids = set()
    if args.resume and os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r['case_id'])
        print(f"  Resuming: {len(done_ids)} already done")

    remaining = [c for c in cases if c['case_id'] not in done_ids]
    print(f"  {len(remaining)} cases to summarize")

    if not remaining:
        print("Nothing to do!")
        return

    llm_call = make_llm_call_fn(args)

    # Summarize
    t0 = time.time()
    with open(OUTPUT_JSONL, 'a') as f:
        for i, case in enumerate(remaining):
            try:
                summary = summarize_case(case, llm_call)
            except Exception as e:
                print(f"  ERROR on {case['case_id']}: {e}")
                summary = ""

            record = {
                'case_id': case['case_id'],
                'source': case['source'],
                'n_paragraphs': case['n_paragraphs'],
                'summary': summary,
                'passage_ids': case['passage_ids'][:5],  # first 5 for reference
            }
            f.write(json.dumps(record) + '\n')

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  {i+1:>6,}/{len(remaining):,} ({(i+1)/len(remaining)*100:.1f}%) | "
                      f"{rate:.1f} cases/sec | ETA {eta/60:.0f}min | "
                      f"summary: {summary[:80]}...")

    total_time = time.time() - t0
    print(f"\nDone! {len(remaining)} summaries in {total_time/60:.1f}min ({total_time/3600:.1f}hr)")
    print(f"Output: {OUTPUT_JSONL}")

    # Also write CSV for easy inspection
    if os.path.exists(OUTPUT_JSONL):
        records = []
        with open(OUTPUT_JSONL) as f:
            for line in f:
                records.append(json.loads(line))
        pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
        print(f"CSV: {OUTPUT_CSV} ({len(records)} rows)")


if __name__ == "__main__":
    main()
