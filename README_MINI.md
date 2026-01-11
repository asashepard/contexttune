# Quick Start: SWE-bench Sanity Check

## Install Dependencies

```bash
pip install swebench
```

## Run Sanity Check

```bash
./scripts/sanity_one_instance.sh django__django-16379
```

## Output Locations

- **Predictions**: `artifacts/preds/<run_id>/preds.jsonl`
- **Results**: `results/<run_id>/` (stdout.log, stderr.log, cmd.txt)

## Custom Dataset

```bash
./scripts/sanity_one_instance.sh django__django-16379 princeton-nlp/SWE-bench_Lite
```
