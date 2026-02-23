"""Download the barexam_qa dataset from HuggingFace.

Usage:
  uv run python download_data.py          # Download passages + QA splits
  uv run python download_data.py --check  # Check if data already exists

Downloads from: https://huggingface.co/datasets/reglab/barexam_qa
Two configs: "passages" (~686K bar exam passages) and "qa" (~1.2K QA pairs).
"""

import os
import sys

DATA_DIR = "datasets/barexam_qa"
PASSAGES_CSV = os.path.join(DATA_DIR, "barexam_qa_train.csv")
QA_CSV = os.path.join(DATA_DIR, "qa", "qa.csv")


def check_data():
    """Check if required data files exist."""
    files = {
        "Passages CSV": PASSAGES_CSV,
        "QA CSV": QA_CSV,
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
    """Download barexam_qa from HuggingFace and save to datasets/barexam_qa/."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Run: uv sync")
        sys.exit(1)

    import pandas as pd

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "qa"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "passages"), exist_ok=True)

    # --- Passages (the main retrieval corpus) ---
    print("Downloading passages from reglab/barexam_qa (passages config)...")
    passages = load_dataset("reglab/barexam_qa", "passages")

    for split_name, split_data in passages.items():
        if split_name == "train":
            path = PASSAGES_CSV
        else:
            path = os.path.join(DATA_DIR, "passages", f"barexam_qa_{split_name}.csv")
        split_data.to_csv(path, index=False)
        print(f"  {split_name}: {len(split_data)} rows -> {path}")

    # --- QA pairs (questions + gold passage indices) ---
    print("\nDownloading QA pairs from reglab/barexam_qa (qa config)...")
    qa = load_dataset("reglab/barexam_qa", "qa")

    all_splits = []
    for split_name, split_data in qa.items():
        split_path = os.path.join(DATA_DIR, "qa", f"{split_name}.csv")
        split_data.to_csv(split_path, index=False)
        print(f"  {split_name}: {len(split_data)} rows -> {split_path}")
        all_splits.append(pd.DataFrame(split_data))

    # Create combined qa.csv (all splits merged — this is what eval scripts expect)
    combined = pd.concat(all_splits, ignore_index=True)
    combined.to_csv(QA_CSV, index=False)
    print(f"  Combined all splits -> {QA_CSV} ({len(combined)} rows)")

    print(f"\nDone! Files saved to {DATA_DIR}/")
    print("\nNext steps:")
    print("  uv run python load_corpus.py curated   # Fast: ~1.5K passages, ~3 min")
    print("  uv run python load_corpus.py 20000      # Full: 20K passages, ~30 min")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        print("Checking data files:")
        if check_data():
            print("\nAll data files present.")
        else:
            print("\nSome files missing. Run: uv run python download_data.py")
        return

    if os.path.isfile(PASSAGES_CSV):
        print(f"Data already exists at {PASSAGES_CSV}.")
        print("Re-downloading will overwrite existing files.")
        resp = input("Continue? [y/N] ")
        if resp.lower() != "y":
            print("Cancelled.")
            return

    download()


if __name__ == "__main__":
    main()
