import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

DEFAULT_DETECTORS = ["PCA", "IForest", "OCSVM", "ECOD"]


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\\n", "", text)
        text = re.sub(r"\\n```$", "", text)
    return text.strip()


def extract_function(text: str) -> str:
    cleaned = strip_markdown_fence(text)
    marker = "def generate_hard_anomalies"
    idx = cleaned.find(marker)
    if idx == -1:
        return cleaned
    # Keep only from function definition onward.
    return cleaned[idx:].strip()


def query_text(client, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_code_answers(
    base_dir: Path,
    llm_type: str,
    model_name: str,
    detectors: List[str],
    n_queries: int,
    temperature: float,
    max_tokens: int,
    base_url: Optional[str],
    dry_run: bool,
) -> None:
    prompt_dir = base_dir / "prompt" / "4code"
    out_dir = base_dir / "answer" / "code" / llm_type
    ensure_dir(out_dir)

    client = None
    if not dry_run:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Please export it before querying.")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)

    for detector in detectors:
        prompt_path = prompt_dir / f"{detector}.txt"
        if not prompt_path.exists():
            print(f"[skip] Prompt not found: {prompt_path}")
            continue

        prompt = prompt_path.read_text(encoding="utf-8")
        for i in range(1, n_queries + 1):
            out_path = out_dir / f"{detector}_{i}.py"
            if dry_run:
                print(f"[dry-run] would query {detector} q{i} -> {out_path}")
                continue

            raw = query_text(client, model_name, prompt, temperature, max_tokens)
            code = extract_function(raw)
            out_path.write_text(code + "\n", encoding="utf-8")
            print(f"[saved] {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LLM and save hard anomaly code answers.")
    parser.add_argument("--base_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--llm_type", type=str, default="gemini-2.5-pro")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--detectors", type=str, default=",".join(DEFAULT_DETECTORS))
    parser.add_argument("--n_queries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "").strip())
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    generate_code_answers(
        base_dir=Path(args.base_dir),
        llm_type=args.llm_type,
        model_name=args.model,
        detectors=detectors,
        n_queries=args.n_queries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url or None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
