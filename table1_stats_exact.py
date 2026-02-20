import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import stats


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
    candidates = [with_lambda, legacy + ".npy"]
    for name in candidates:
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            continue
        try:
            return float(np.load(path, allow_pickle=True).item().get(metric, np.nan))
        except Exception:
            return np.nan
    return np.nan


def compute_table1_exact(workdir, cfg):
    base_dir = os.path.join(workdir, "results_baseline")
    star_dir = os.path.join(workdir, "results_star")
    rows = []

    for metric in METRICS:
        for model in MODELS:
            x_dataset = []
            y_dataset = []
            effects = 0

            for d in FILE_NAMES:
                x_seed = []
                y_seed = []
                for s in SEEDS:
                    hp = resolve_hp(cfg, d, model)
                    x_seed.append(load_metric(base_dir, d, model, s, hp, metric))
                    y_seed.append(load_metric(star_dir, d, model, s, hp, metric))

                x = np.nanmean(x_seed)
                y = np.nanmean(y_seed)
                if np.isnan(x) or np.isnan(y):
                    continue
                x_dataset.append(x)
                y_dataset.append(y)
                if y > x:
                    effects += 1

            x_arr = np.asarray(x_dataset, dtype=float)
            y_arr = np.asarray(y_dataset, dtype=float)
            improvement = y_arr - x_arr
            relative_improvement = (improvement / x_arr) * 100.0

            t_statistic, p_value = stats.ttest_rel(y_arr, x_arr)
            if t_statistic > 0:
                one_tailed_p = p_value / 2.0
            else:
                one_tailed_p = 1.0 - (p_value / 2.0)

            rows.append({
                "Metric": metric,
                "Model": model,
                "Original_Mean": float(np.mean(x_arr)),
                "Mean_Improvement": float(np.mean(improvement)),
                "Mean_Relative_Improvement_%": float(np.mean(relative_improvement)),
                "Effects": int(effects),
                "Total": int(len(x_arr)),
                "T_Statistic": float(t_statistic),
                "One_Tailed_P_Value_(y>x)": float(one_tailed_p),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--preset", type=str, default="presets/table4_predefine.json")
    args = parser.parse_args()

    preset = args.preset
    if not os.path.isabs(preset):
        preset = os.path.join(args.workdir, preset)
    cfg = load_preset(preset)

    out = compute_table1_exact(args.workdir, cfg)

    # Format to paper-friendly significant digits.
    out_fmt = out.copy()
    out_fmt["Original_Mean"] = out_fmt["Original_Mean"].map(lambda x: f"{x:.4f}")
    out_fmt["Mean_Improvement"] = out_fmt["Mean_Improvement"].map(lambda x: f"{x:.4f}")
    out_fmt["Mean_Relative_Improvement_%"] = out_fmt["Mean_Relative_Improvement_%"].map(lambda x: f"{x:.2f}")
    out_fmt["T_Statistic"] = out_fmt["T_Statistic"].map(lambda x: f"{x:.4f}")
    out_fmt["One_Tailed_P_Value_(y>x)"] = out_fmt["One_Tailed_P_Value_(y>x)"].map(lambda x: f"{x:.4g}")

    out_path = os.path.join(args.workdir, "table1_style_stats_exact.csv")
    out_fmt.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out_fmt.to_string(index=False))


if __name__ == "__main__":
    main()
