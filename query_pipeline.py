import argparse
import os
from pathlib import Path
from typing import List, Optional

from get_answer import DEFAULT_DETECTORS, generate_code_answers


CODE_PROMPT_TEMPLATE = """You are an expert in anomaly detection systems. The training set contains only normal samples. We use a {detector} detector, where the anomaly score is computed using model.predict_score(). The higher the score, the more anomalous the sample.

# The description of {detector}.
{description}

# Objective
Your task is to write a Python function generate_hard_anomalies(...) that generates anomalies which are the most difficult for the {detector} detector to detect. This means that the generated anomalies should have relatively low anomaly score, thus they are hard to be detected. But these anomalies are helpful to build a more robust detector. After the Python function is completed, users can provide the function with:

* A trained {detector} model (model) that exposes predict_score(),

* The training samples (X_train)

# Requirements:
Your should strictly follow below requirements:

1. You must use your expertise to give anomalies generation policies that are specifically designed for {detector}, not a model-agnostic policy.

2. Generated samples should have as low a score as possible from model.predict_score(). To achieve it, you can first find the set of borderline normal training samples based on your unique and professional understanding of {detector}, not only based on the anomaly score. Then transform them to anomalies that are tailor-designed for {detector}. Please note that the transformation should be specific to {detector}, not a general transformation for other detectors.

3. For the model, you can only use the function model.predict_score.

4. Use NumPy to generate the samples, and output an array of shape (n_samples, d). It should generate as many anomalies as requested.

5. The function should allow setting:
   * the number of samples (n_samples),
   * the trained {detector} model (model),
   * training samples (X_train).

   Thus the function format is generate_hard_anomalies(n_samples: int, model, X_train: np.ndarray)

6. All package imports must be done inside the function.

Return only the complete Python function generate_hard_anomalies(...), with the policy used for generating anomalies and clear comments explaining key steps.
"""


DESCRIPTION_PROMPT_TEMPLATE = (
    "You are an expert in anomaly detection systems. The training set contains only normal samples. "
    "I will provide you the <{detector}>. Please return a description of this detector, and the pseudo code of the algorithm (step1, step2,...). "
    "The answer format is: first one paragraph description; then a bullet list of the main algorithm steps."
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def query_text(client, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def build_client(base_url: Optional[str], dry_run: bool):
    if dry_run:
        return None
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it before querying.")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def generate_descriptions(
    base_dir: Path,
    detectors: List[str],
    model_name: str,
    temperature: float,
    max_tokens: int,
    base_url: Optional[str],
    dry_run: bool,
) -> None:
    prompt_dir = base_dir / "prompt" / "4description"
    out_dir = base_dir / "answer" / "description"
    ensure_dir(prompt_dir)
    ensure_dir(out_dir)

    client = build_client(base_url, dry_run)

    for detector in detectors:
        custom_prompt_path = prompt_dir / f"{detector}.txt"
        if custom_prompt_path.exists():
            prompt = custom_prompt_path.read_text(encoding="utf-8")
        else:
            prompt = DESCRIPTION_PROMPT_TEMPLATE.format(detector=detector)

        out_path = out_dir / f"{detector}.txt"
        if dry_run:
            print(f"[dry-run] would query description for {detector} -> {out_path}")
            continue

        text = query_text(client, model_name, prompt, temperature, max_tokens)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"[saved] {out_path}")


def generate_code_prompts(base_dir: Path, detectors: List[str], dry_run: bool) -> None:
    desc_dir = base_dir / "answer" / "description"
    prompt_dir = base_dir / "prompt" / "4code"
    ensure_dir(prompt_dir)

    for detector in detectors:
        desc_path = desc_dir / f"{detector}.txt"
        if not desc_path.exists():
            print(f"[skip] Description not found: {desc_path}")
            continue

        description = desc_path.read_text(encoding="utf-8").strip()
        prompt = CODE_PROMPT_TEMPLATE.format(detector=detector, description=description)
        out_path = prompt_dir / f"{detector}.txt"
        if dry_run:
            print(f"[dry-run] would write code prompt -> {out_path}")
            continue
        out_path.write_text(prompt + "\n", encoding="utf-8")
        print(f"[saved] {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate LLM-DAS query pipeline.")
    parser.add_argument("--base_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--llm_type", type=str, default="gemini-2.5-pro")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--detectors", type=str, default=",".join(DEFAULT_DETECTORS))
    parser.add_argument("--n_queries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "").strip())
    parser.add_argument("--skip_description_query", action="store_true")
    parser.add_argument("--skip_code_query", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    if not args.skip_description_query:
        generate_descriptions(
            base_dir=base_dir,
            detectors=detectors,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url or None,
            dry_run=args.dry_run,
        )

    generate_code_prompts(base_dir=base_dir, detectors=detectors, dry_run=args.dry_run)

    if not args.skip_code_query:
        generate_code_answers(
            base_dir=base_dir,
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
