from importlib.util import find_spec
import sys

REQUIRED_BASE = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "pyod",
    "tqdm",
]
REQUIRED_QUERY = ["openai"]


def check(modules, title):
    print(f"\n[{title}]")
    missing = []
    for m in modules:
        if find_spec(m) is None:
            print(f"- {m}: MISSING")
            missing.append(m)
        else:
            print(f"- {m}: OK")
    return missing


def main():
    print(f"Python: {sys.version.split()[0]}")
    base_missing = check(REQUIRED_BASE, "Base Reproduction Dependencies")
    query_missing = check(REQUIRED_QUERY, "Query Pipeline Dependencies")

    if not base_missing:
        print("\nBase reproduction environment: READY")
    else:
        print(f"\nBase reproduction environment: NOT READY (missing: {', '.join(base_missing)})")

    if not query_missing:
        print("Query pipeline environment: READY")
    else:
        print(f"Query pipeline environment: NOT READY (missing: {', '.join(query_missing)})")


if __name__ == "__main__":
    main()
