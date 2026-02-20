import argparse
import json
import os
import pandas as pd


def parse_avg_row(csv_path):
    df = pd.read_csv(csv_path)
    row = df[df["Dataset"] == "Average"]
    if row.empty:
        raise ValueError(f"Average row not found in {csv_path}")
    row = row.iloc[0]
    out = {}
    for k, v in row.items():
        if k == "Dataset":
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--targets", type=str, default="presets/paper_table4_targets_avg.json")
    args = parser.parse_args()

    target_path = args.targets
    if not os.path.isabs(target_path):
        target_path = os.path.join(args.workdir, target_path)

    with open(target_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    pr = parse_avg_row(os.path.join(args.workdir, "table_llmdas_AUC-PR_preset.csv"))
    roc = parse_avg_row(os.path.join(args.workdir, "table_llmdas_AUC-ROC_preset.csv"))

    print("AUC-PR (ours - paper):")
    for k, t in targets["AUC-PR"].items():
        o = pr.get(k, float("nan"))
        print(f"{k}: ours={o:.3f}, paper={t:.3f}, gap={o-t:+.3f}")

    print("\nAUC-ROC (ours - paper):")
    for k, t in targets["AUC-ROC"].items():
        o = roc.get(k, float("nan"))
        print(f"{k}: ours={o:.3f}, paper={t:.3f}, gap={o-t:+.3f}")


if __name__ == "__main__":
    main()

