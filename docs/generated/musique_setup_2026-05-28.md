# MuSiQue Setup - 2026-05-28

Normalized Hugging Face `dgslibisey/MuSiQue`, split `validation`, for per-question dense retrieval over the provided candidate paragraphs. No files under `paper/` were edited.

| Field | Value |
|---|---:|
| Questions | 2417 |
| Candidate paragraphs | 48315 |
| Filtered unanswerable rows | 0 |
| Filtered rows without gold support | 0 |

Gold paragraph count distribution: `{"2": 1252, "3": 760, "4": 405}`.

Candidate paragraph count distribution: `{"17": 2, "18": 5, "19": 9, "20": 2401}`.

Hop-count distribution: `{"2": 1252, "3": 760, "4": 405}`.

Outputs:
- `datasets/musique/questions.csv`
- `datasets/musique/passages.csv`
