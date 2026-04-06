#!/usr/bin/env bash
set -euo pipefail

EMBED_SHORT="${1:-stella-400m}"
PROVIDER="${2:-or-gemma27b}"
MODE="${3:-rag_simple}"
QUESTIONS="${4:-full}"
DATASET="${5:-barexam}"

REPO="/home/techguy227/grad/LegalRagAgent"
cd "$REPO"

STATUS_LOG="logs/embed_probe_${EMBED_SHORT}_${PROVIDER}_${MODE}_${QUESTIONS}_${DATASET}.status"
mkdir -p logs scripts/hpc

echo "[$(date '+%F %T')] Starting embedding probe" | tee -a "$STATUS_LOG"
echo "embedding=$EMBED_SHORT provider=$PROVIDER mode=$MODE questions=$QUESTIONS dataset=$DATASET" | tee -a "$STATUS_LOG"

# Step 1: embed into a model-specific collection. Non-default embedding models
# auto-suffix the collection name, so the baseline collection is not overwritten.
uv run python utils/fast_embed.py "$DATASET" --model "$EMBED_SHORT" >> "$STATUS_LOG" 2>&1

echo "[$(date '+%F %T')] Embedding complete; resolving collection name" | tee -a "$STATUS_LOG"
COLLECTION_NAME=$(EMBED_SHORT="$EMBED_SHORT" uv run python - <<'PY'
from utils.fast_embed import EMBEDDING_MODELS, resolve_collection_name
import os
short=os.environ['EMBED_SHORT']
model_id=EMBEDDING_MODELS.get(short, short)
print(resolve_collection_name('legal_passages', model_id))
PY
)
MODEL_ID=$(EMBED_SHORT="$EMBED_SHORT" uv run python - <<'PY'
from utils.fast_embed import EMBEDDING_MODELS
import os
short=os.environ['EMBED_SHORT']
print(EMBEDDING_MODELS.get(short, short))
PY
)

echo "collection=$COLLECTION_NAME model_id=$MODEL_ID" | tee -a "$STATUS_LOG"

# Step 2: eval with env override, keeping the base collection untouched.
EVAL_COLLECTION_OVERRIDE="$COLLECTION_NAME" \
EVAL_EMBEDDING_MODEL="$MODEL_ID" \
uv run python eval/eval_harness.py \
  --mode "$MODE" \
  --provider "$PROVIDER" \
  --questions "$QUESTIONS" \
  --dataset "$DATASET" \
  --tag "embed-${EMBED_SHORT}-${PROVIDER}-${MODE}" >> "$STATUS_LOG" 2>&1

echo "[$(date '+%F %T')] Eval complete; committing experiments.jsonl if changed" | tee -a "$STATUS_LOG"
git add -f logs/experiments.jsonl
if ! git diff --cached --quiet; then
  git commit -m "data(eval): log ${MODE}/${PROVIDER} with ${EMBED_SHORT} embedding" >> "$STATUS_LOG" 2>&1 || true
  git push origin HEAD >> "$STATUS_LOG" 2>&1 || true
else
  echo "[$(date '+%F %T')] No experiments.jsonl diff to commit" >> "$STATUS_LOG"
fi

echo "[$(date '+%F %T')] Finished embedding probe" | tee -a "$STATUS_LOG"
