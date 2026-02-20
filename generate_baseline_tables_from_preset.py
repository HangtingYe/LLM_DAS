import argparse
import json
import os
import numpy as np
import pandas as pd


FILE_NAMES = [
    "Cardiotocography", "Hepatitis", "Parkinson", "SpamBase", "WDBC", "WPBC",
    "Wilt", "abalone", "amazon", "annthyroid", "arrhythmia", "breastw",
    "cardio", "comm.and.crime", "fault", "glass", "imgseg", "ionosphere",
    "lympho", "mammography", "mnist", "musk", "optdigits", "pendigits", "pima",
    "satellite", "satimage-2", "shuttle", "speech", "thyroid", "vertebral",
    "vowels", "wbc", "wine", "yeast", "campaign",
]
MODELS = ["PCA", "IForest", "OCSVM", "ECOD"]
SEEDS = [42, 0, 100, 17, 21]
METRICS = ["AUC-PR", "AUC-ROC"]


def load_preset(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "models" not in cfg:
        raise ValueError("preset json must contain field 'models'")
    return cfg


def resolve_hp(cfg, dataname, model):
    defaults = cfg.get("defaults", {})
    mcfg = cfg.get("models", {}).get(model, {})
    dcfg = mcfg.get("datasets", {}).get(dataname, {})

    def pick(key, fallback):
        return dcfg.get(key, mcfg.get(key, defaults.get(key, fallback)))

    lam = float(pick("lambda_score", 1.0))
    if lam <= 0:
        lam = 1e-4
    return {
        "query": int(pick("query", 1)),
        "llm_type": str(pick("llm_type", "gemini-2.5-pro")),
        "shuffle": str(pick("shuffle", "False")),
        "classifier": str(pick("classifier", "rf")),
        "lambda_score": lam,
    }


def load_metric(results_dir, dataname, model, seed, hp, metric):
    legacy = f"{dataname}_{model}_none_{seed}_{hp['query']}_{hp['llm_type']}_{hp['shuffle']}_{hp['classifier']}"
    with_lambda = f"{legacy}_{hp['lambda_score']:.6g}.npy"
    for name in [with_lambda, legacy + ".npy"]:
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            continue
        try:
            return float(np.load(path, allow_pickle=True).item().get(metric, np.nan))
        except Exception:
            return np.nan
    return np.nan


def fmt_mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, "NaN"
    m, s = float(np.mean(arr)), float(np.std(arr))
    return m, f"{m:.3f}\u00b1{s:.3f}"


def build_baseline_table(workdir, cfg, metric):
    rows = []
    baseline_dir = os.path.join(workdir, "results_baseline")
    mean_tracker = {m: [] for m in MODELS}

    for d in sorted(FILE_NAMES, key=str.lower):
        row = {"Dataset": d}
        for m in MODELS:
            vals = []
            for s in SEEDS:
                hp = resolve_hp(cfg, d, m)
                vals.append(load_metric(baseline_dir, d, m, s, hp, metric))
            mean_v, text = fmt_mean_std(vals)
            row[m] = text
            if not np.isnan(mean_v):
                mean_tracker[m].append(mean_v)
        rows.append(row)

    df = pd.DataFrame(rows)
    avg = {"Dataset": "Average"}
    for m in MODELS:
        a = np.asarray(mean_tracker[m], dtype=float)
        avg[m] = f"{np.nanmean(a):.3f}" if a.size > 0 else "NaN"
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--preset", type=str, default="presets/table4_predefine.json")
    args = parser.parse_args()

    preset = args.preset
    if not os.path.isabs(preset):
        preset = os.path.join(args.workdir, preset)
    cfg = load_preset(preset)

    for metric in METRICS:
        df = build_baseline_table(args.workdir, cfg, metric)
        out = os.path.join(args.workdir, f"table_baseline_{metric}_preset.csv")
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
