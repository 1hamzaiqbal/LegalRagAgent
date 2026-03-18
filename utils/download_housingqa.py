"""Download the HousingQA dataset from HuggingFace.

Downloads statute corpus and QA pairs from reglab/housing_qa.
Saves to datasets/housing_qa/ in CSV format for compatibility with load_corpus.py.

Usage:
  uv run python utils/download_housingqa.py          # Download all splits
  uv run python utils/download_housingqa.py --check  # Check if data exists
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "datasets/housing_qa"
STATUTES_CSV = os.path.join(DATA_DIR, "statutes.csv")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions.csv")


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
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Statute corpus (retrieval passages) ---
    print("Downloading statute corpus from reglab/housing_qa...")
    statutes = load_dataset("reglab/housing_qa", "statutes", split="corpus")
    print(f"  Loaded {len(statutes)} statutes")

    # Convert to DataFrame and save as CSV
    # Rename fields to match barexam_qa format: idx, text, source
    df_statutes = statutes.to_pandas()
    df_statutes = df_statutes.rename(columns={"text": "text"})
    # Add source column for compatibility
    df_statutes["source"] = "housing_statute"
    df_statutes.to_csv(STATUTES_CSV, index=False)
    print(f"  Saved {len(df_statutes)} statutes to {STATUTES_CSV}")

    # --- QA pairs (with gold statute labels) ---
    print("\nDownloading QA pairs from reglab/housing_qa...")
    questions = load_dataset("reglab/housing_qa", "questions", split="test")
    print(f"  Loaded {len(questions)} QA pairs")

    df_questions = questions.to_pandas()

    # Extract gold statute indices from the nested structure
    def extract_gold_idxs(statutes_list):
        if not statutes_list:
            return ""
        return ",".join(str(s["statute_idx"]) for s in statutes_list if "statute_idx" in s)

    df_questions["gold_idx"] = df_questions["statutes"].apply(extract_gold_idxs)
    df_questions.to_csv(QUESTIONS_CSV, index=False)
    print(f"  Saved {len(df_questions)} QA pairs to {QUESTIONS_CSV}")

    print(f"\nDone! Files saved to {DATA_DIR}/")
    print(f"\nDataset summary:")
    print(f"  Statutes (corpus):  {len(df_statutes):,} passages")
    print(f"  Questions (QA):     {len(df_questions):,} pairs")
    print(f"  States covered:     {df_statutes['state'].nunique()}")
    print(f"\nNext step:")
    print(f"  uv run python utils/load_corpus.py housing 200000  # Load 200K statutes")


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
