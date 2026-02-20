# LLM_DAS (LLM-DAS Reproduction)

This repository reproduces baseline and `base + LLM-DAS` results (Table 4 style) on tabular anomaly detection datasets.

## 0. Read This First: Which path should I run?

You have two valid workflows:

1. **Path A (recommended for reproduction users)**: use existing query outputs already stored in this repo.
2. **Path B (for method users)**: generate your own query outputs with API, then run Path A.

Decision guide:

- If you only want to reproduce reported tables quickly: **run Path A only**.
- If you want your own LLM-generated anomaly code: **run Path B first**, then **run Path A**.

Important relationships (for Path B steps only):

- If you run **3.2 full pipeline**, you do **not** need 3.3.
- If you run **3.3 code-only query**, you do **not** need 3.2 (but your `prompt/4code/*.txt` must already be ready).
- Section 4 (`dry_run`) is always optional sanity check.
- After finishing Path B (either 3.2 or 3.3), you should go to Path A Section 2 to run experiments and generate tables.

## 1. Environment Setup

Recommended for readers: create environment from `environment.yml` (versions aligned with the author's validated stack).

```bash
cd LLM_DAS

conda env remove -n llm_das -y 2>/dev/null || true
conda env create -f environment.yml
conda activate llm_das

# Environment check
python check_env.py
```

Expected from `python check_env.py`:
- Base reproduction dependencies should all be `OK` (especially `torch`).
- `openai` is only required for Path B.

If you see `No matching distribution found`:
- Usually this is an index/network issue, not package absence.
- Check with `python -m pip config list`.
- Then retry with an explicit index such as `-i https://pypi.org/simple` or your regional mirror.

Recommended reproducibility settings before running experiments:

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## 2. Path A: Reproduce with existing query outputs

This is the default path for most readers.

### 2.1 Run experiments

```bash
# optional: clean old outputs before a fully fresh rerun
rm -rf results_baseline results_star results_time_baseline results_time_star models_baseline models_star hard_anomalies

python run_job_baseline.py --workers 16 --preset presets/table4_predefine.json
```

Notes:
- Choose `--workers` based on machine safety (e.g., 8/16/32).

### 2.2 Generate result tables

```bash
# LLM-DAS (base + LLM-DAS)
python generate_table4_from_preset.py --preset presets/table4_predefine.json

# Baseline
python generate_baseline_tables_from_preset.py --preset presets/table4_predefine.json

# Table 1 statistics (paper-style significant digits)
python table1_stats_exact.py --preset presets/table4_predefine.json
```

Generated files:
- `table_llmdas_AUC-PR_preset.csv`
- `table_llmdas_AUC-ROC_preset.csv`
- `table_baseline_AUC-PR_preset.csv`
- `table_baseline_AUC-ROC_preset.csv`
- `table1_style_stats_exact.csv`

## 3. Path B: Regenerate query outputs by yourself

Use this if you do not want to rely on provided query outputs.

### 3.1 Configure API credentials (required for real query)

```bash
export OPENAI_API_KEY="your_api_key"
# optional for compatible endpoint
# export OPENAI_BASE_URL="https://your-endpoint/v1"
```

Never hardcode keys in source files.

### 3.2 Full automated pipeline (description -> prompt -> code)  

```bash
python query_pipeline.py \
  --llm_type gemini-2.5-pro \
  --model gemini-2.5-pro \
  --detectors PCA,IForest,OCSVM,ECOD \
  --n_queries 3
```

What it does:
- query detector descriptions -> `answer/description/<DETECTOR>.txt`
- build code prompts -> `prompt/4code/<DETECTOR>.txt`
- query hard-anomaly code -> `answer/code/<llm_type>/<DETECTOR>_{1..n}.py`

After 3.2 finishes, go to **Path A Section 2** to run experiments and generate tables.

### 3.3 Code-only query (skip description stage)

Use this when `prompt/4code/*.txt` are already prepared.

```bash
python get_answer.py \
  --llm_type gemini-2.5-pro \
  --model gemini-2.5-pro \
  --detectors PCA,IForest,OCSVM,ECOD \
  --n_queries 3
```

After 3.3 finishes, go to **Path A Section 2**.

## 4. Optional sanity checks (`dry_run`, no API call)

`dry_run` is optional and does not replace real execution.

```bash
python query_pipeline.py --dry_run
python get_answer.py --dry_run
```

## 5. Common command patterns

Re-run all jobs even if output already exists:

```bash
python run_job_baseline.py --workers 16 --rerun_all --preset presets/table4_predefine.json
```

Use a custom preset:

```bash
python run_job_baseline.py --workers 16 --preset presets/your_preset.json
```

## 6. Portability notes

- All project paths are relative; no absolute machine-specific path is required.
- The code is designed to be reusable across devices with the same folder structure.
- Results can vary across devices/environments due to numerical/runtime differences. In reproduction, the key criterion is whether `base + LLM-DAS` consistently improves over baseline, which supports the effectiveness of LLM-DAS.
