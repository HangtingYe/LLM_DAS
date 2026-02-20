#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys

CORE = [
    "python",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyod",
    "torch",
    "tqdm",
    "requests",
    "fsspec",
    "matplotlib",
    "openai",
]


def conda_list(env_name: str):
    cmd = ["conda", "list", "-n", env_name, "--json"]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    return {pkg["name"]: pkg["version"] for pkg in data}


def main():
    parser = argparse.ArgumentParser(description="Compare package versions between two conda envs.")
    parser.add_argument("--base_env", default="base")
    parser.add_argument("--target_env", default="llm_das")
    args = parser.parse_args()

    try:
        base = conda_list(args.base_env)
        tgt = conda_list(args.target_env)
    except Exception as e:
        print(f"Failed to query conda envs: {e}")
        sys.exit(2)

    mismatches = []
    missing = []
    for name in CORE:
        bv = base.get(name)
        tv = tgt.get(name)
        if bv is None:
            continue
        if tv is None:
            missing.append((name, bv))
        elif tv != bv:
            mismatches.append((name, bv, tv))

    if not mismatches and not missing:
        print(f"PASS: {args.target_env} is aligned with {args.base_env} for core packages.")
        return

    print(f"FAIL: {args.target_env} is not aligned with {args.base_env}.")
    if missing:
        print("Missing in target:")
        for n, bv in missing:
            print(f"- {n}: base={bv}, target=MISSING")
    if mismatches:
        print("Version mismatches:")
        for n, bv, tv in mismatches:
            print(f"- {n}: base={bv}, target={tv}")
    sys.exit(1)


if __name__ == "__main__":
    main()
