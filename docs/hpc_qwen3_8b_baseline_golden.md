# HPC Qwen3-8B Baseline + Golden Eval (WashU Cluster)

Recommended first local-model cluster experiment for LegalRAG.

## Why this specific run
This path avoids embedding/retrieval setup and just asks:
1. how good is local **Qwen3-8B** with base knowledge only?
2. how much does **golden passage** help the same local model?

That makes it the cleanest first SSH/HPC bring-up for another agent or operator.

## Required repo resources
- `scripts/hpc/slurm_vllm_eval_qwen3_8b_baseline_golden.sh`
- `docs/hpc_qwen3_8b_eval.md`
- `docs/cluster_workflow.md`

## What the job does
- starts local vLLM for `Qwen/Qwen3-8B`
- runs `llm_only / full / barexam`
- then runs `golden_passage / full / barexam`
- uses the same model server for both runs
- requires **no embedding rebuild** and **no retriever setup**

## Launch
```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
sbatch scripts/hpc/slurm_vllm_eval_qwen3_8b_baseline_golden.sh
```

## Monitor
```bash
squeue -u hiqbal
tail -f /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/<jobid>.out
tail -f /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/vllm_qwen3_8b_<jobid>.log
```

## Success condition
A good first cluster result means you get both of these new logs:
- `llm_only / cluster-vllm / full / barexam`
- `golden_passage / cluster-vllm / full / barexam`

Once that works, you can decide whether to:
- add `rag_simple`
- test a stronger 8B-14B family model
- or move to retrieval/embedding-dependent cluster experiments later
