"""Download the HousingQA dataset from HuggingFace.

Downloads statute corpus and QA pairs directly from reglab/housing_qa.
Saves to datasets/housing_qa/ in CSV format.

Usage:
  uv run python utils/download_housingqa.py          # Download all splits
  uv run python utils/download_housingqa.py --check  # Check if data exists
"""

import os
import sys
import zipfile
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "datasets/housing_qa"
STATUTES_CSV = os.path.join(DATA_DIR, "statutes.csv")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions.csv")

REPO_BASE = "https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data"


def check_data():
    """Check if required data files exist."""
    files = {
        "Statutes CSV": STATUTES_CSV,
        "Questions CSV": QUESTIONS_CSV,
    }
    all_ok = True
    for label, path in files.items():
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        print(f"  {status}: {label} ({path})")
        if not exists:
            all_ok = False
    return all_ok


def download():
    """Download HousingQA from HuggingFace and save to datasets/housing_qa/."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Statutes (the retrieval corpus) ---
    print("Downloading statutes corpus from reglab/housing_qa...")
    zip_path = hf_hub_download(
        repo_id="reglab/housing_qa",
        filename="data/statutes.tsv.zip",
        repo_type="dataset",
    )

    print("  Extracting statutes TSV...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the TSV file inside the zip
        tsv_names = [n for n in zf.namelist() if n.endswith('.tsv')]
        if not tsv_names:
            raise RuntimeError(f"No TSV found in zip: {zf.namelist()}")
        tsv_name = tsv_names[0]
        zf.extract(tsv_name, DATA_DIR)
        extracted_path = os.path.join(DATA_DIR, tsv_name)

    print("  Converting TSV to CSV...")
    df_statutes = pd.read_csv(extracted_path, sep='\t')
    # Add source column for pipeline compatibility
    df_statutes["source"] = "housing_statute"
    df_statutes.to_csv(STATUTES_CSV, index=False)
    print(f"  Saved {len(df_statutes):,} statutes to {STATUTES_CSV}")

    # Cleanup extracted TSV
    if os.path.exists(extracted_path) and extracted_path != STATUTES_CSV:
        os.remove(extracted_path)

    # --- Questions (QA pairs with gold statute labels) ---
    print("\nDownloading QA pairs from reglab/housing_qa...")
    q_zip_path = hf_hub_download(
        repo_id="reglab/housing_qa",
        filename="data/questions.json.zip",
        repo_type="dataset",
    )

    print("  Extracting questions JSON...")
    with zipfile.ZipFile(q_zip_path, 'r') as zf:
        json_names = [n for n in zf.namelist() if n.endswith('.json')]
        json_name = json_names[0]
        zf.extract(json_name, DATA_DIR)
        extracted_json = os.path.join(DATA_DIR, json_name)

    with open(extracted_json, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    # Flatten the nested structure to a DataFrame
    rows = []
    for q in questions_data:
        gold_idxs = []
        if q.get("statutes"):
            gold_idxs = [str(s["statute_idx"]) for s in q["statutes"] if "statute_idx" in s]
        rows.append({
            "idx": q["idx"],
            "state": q.get("state", ""),
            "question": q.get("question", ""),
            "answer": q.get("answer", ""),
            "gold_idx": ",".join(gold_idxs),
            "question_group": q.get("question_group", 0),
        })

    df_questions = pd.DataFrame(rows)
    df_questions.to_csv(QUESTIONS_CSV, index=False)
    print(f"  Saved {len(df_questions):,} QA pairs to {QUESTIONS_CSV}")

    # Cleanup extracted JSON
    if os.path.exists(extracted_json) and extracted_json != QUESTIONS_CSV:
        os.remove(extracted_json)

    print(f"\nDone! Files saved to {DATA_DIR}/")
    print(f"\nDataset summary:")
    print(f"  Statutes (corpus):  {len(df_statutes):,} passages")
    print(f"  Questions (QA):     {len(df_questions):,} pairs")
    if "state" in df_statutes.columns:
        print(f"  States covered:     {df_statutes['state'].nunique()}")
    print(f"\nNext step:")
    print(f"  uv run python utils/load_corpus.py housing    # Embed all statutes")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        print("Checking HousingQA data files:")
        if check_data():
            print("\nAll data files present.")
        else:
            print("\nSome files missing. Run: uv run python utils/download_housingqa.py")
        return

    if os.path.isfile(STATUTES_CSV):
        print(f"Data already exists at {STATUTES_CSV}.")
        resp = input("Re-download? [y/N] ")
        if resp.lower() != "y":
            print("Cancelled.")
            return

    download()


if __name__ == "__main__":
    main()
