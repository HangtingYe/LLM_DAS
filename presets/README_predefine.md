# Predefined Hyperparameters (`run_job_baseline.py`)

Use `--preset` to run with fixed per-model/per-dataset hyperparameters.

## Run

```bash
python run_job_baseline.py \
  --workers 16 \
  --preset presets/table4_predefine.json
```

Notes:
- Preset paths are workspace-relative.
- `lambda_score` is always forced to be strictly positive (`<=0` is auto-clamped to `1e-4`).

## Generate LLM-DAS Tables

```bash
python generate_table4_from_preset.py \
  --preset presets/table4_predefine.json
```

Outputs:
- `table_llmdas_AUC-PR_preset.csv`
- `table_llmdas_AUC-ROC_preset.csv`

## Generate Baseline Tables

```bash
python generate_baseline_tables_from_preset.py \
  --preset presets/table4_predefine.json
```

Outputs:
- `table_baseline_AUC-PR_preset.csv`
- `table_baseline_AUC-ROC_preset.csv`

## Generate Table 1 Statistics

```bash
python table1_stats_exact.py \
  --preset presets/table4_predefine.json
```

Output:
- `table1_style_stats_exact.csv`
