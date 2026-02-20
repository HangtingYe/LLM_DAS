import os
import time
import shlex
import subprocess
import itertools
import glob
import sys
import argparse
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# datanames = file_names = [
#  'Cardiotocography',
#  'Hepatitis',
#  'Parkinson',
#  'SpamBase',
#  'WDBC',
#  'WPBC',           
#  'Wilt',
#  'abalone',
#  'amazon',
#  'annthyroid',
#  'arrhythmia',
#  'breastw',
#  'census',
#  'cardio',
#  'comm.and.crime',
#  'cover',
#  'fault',
#  'glass',
#  'imgseg',
#  'ionosphere',
# #  'letter',
#  'lympho',
#  'mammography',
#  'mnist',
#  'musk',
#  'optdigits',
#  'pendigits',
#  'pima',
#  'satellite',
#  'satimage-2',
#  'shuttle',
#  'speech',
#  'thyroid',
#  'vertebral',
#  'vowels',
#  'wbc',
#  'wine',
#  'yeast',
#  'backdoor',
#  'fraud', 
#  'campaign'
# ]
datanames = file_names = [
 'Cardiotocography',
 'Hepatitis',
 'Parkinson',
 'SpamBase',
 'WDBC',
 'WPBC',           
 'Wilt',
 'abalone',
 'amazon',
 'annthyroid',
 'arrhythmia',
 'breastw',
#  'census',
 'cardio',
 'comm.and.crime',
#  'cover',
 'fault',
 'glass',
 'imgseg',
 'ionosphere',
#  'letter',
 'lympho',
 'mammography',
 'mnist',
 'musk',
 'optdigits',
 'pendigits',
 'pima',
 'satellite',
 'satimage-2',
 'shuttle',
 'speech',
 'thyroid',
 'vertebral',
 'vowels',
 'wbc',
 'wine',
 'yeast',
#  'backdoor',
#  'fraud', 
 'campaign'
]

gpu_cnt = 1

processors = ['none']

seeds = [42,0,100,17,21]

querys = [1,2,3]

llm_types = ['gemini-2.5-pro']

shuffles = ['False']

classifiers = ['rf']
score_lambdas = [1.0]

# submitting experiments in parallel to multiple gpus
def run(cmds, cuda_id, gpu_cnt):
    _cur = 0

    def recycle_devices():
        running_jobs = 0
        for cid in cuda_id:
            if cuda_id[cid] is not None:
                proc = cuda_id[cid]
                if proc.poll() is not None:
                    cuda_id[cid] = None
                else:
                    running_jobs += 1
        return running_jobs

    def available_device_id():
        for cid in cuda_id:
            if cuda_id[cid] is None:
                return cid

    def submit(cmd, cid):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(int(cid) % gpu_cnt)

        print('Submit Job:')
        print(cmd)

        cmd_args = shlex.split(cmd)
        #log_file = open(f'log/{cmd_args[-1]}', 'w')

        proc = subprocess.Popen(cmd_args, env=env, cwd=BASE_DIR)

        cuda_id[cid] = proc

    while 1:
        running_jobs = recycle_devices()

        if _cur >= len(cmds) and running_jobs == 0:
            break

        while _cur < len(cmds):
            cid = available_device_id()
            if cid is None:
                break
            print('CUDA {} available'.format(cid))
            submit(cmds[_cur], cid)
            _cur += 1

        time.sleep(0.2)


def has_result(dataname, model_type, preprocess, seed, query, llm_type, shuffle, classifier, lambda_score):
    result_file = os.path.join(
        BASE_DIR,
        "results_star",
        f"{dataname}_{model_type}_{preprocess}_{seed}_{query}_{llm_type}_{shuffle}_{classifier}_{float(lambda_score):.6g}.npy",
    )
    return os.path.exists(result_file)


def load_preset_config(preset_path):
    with open(preset_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "models" not in cfg or not isinstance(cfg["models"], dict):
        raise ValueError("preset json must contain object field: models")
    return cfg


def resolve_job_hparams(cfg, dataname, model_type):
    defaults = cfg.get("defaults", {})
    model_cfg = cfg.get("models", {}).get(model_type, {})
    dataset_cfg = model_cfg.get("datasets", {}).get(dataname, {})

    def pick(key, fallback):
        return dataset_cfg.get(key, model_cfg.get(key, defaults.get(key, fallback)))

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


def build_cmd(script_path, dataname, model_type, seed, hp):
    cmd_parts = [
        f'{sys.executable} {script_path}',
        f'--dataname {dataname}',
        f'--model_type {model_type}',
        f'--seed {seed}',
        f'--query {hp["query"]}',
        f'--llm_type {hp["llm_type"]}',
        f'--shuffle {hp["shuffle"]}',
        f'--classifier {hp["classifier"]}',
        f'--lambda_score {hp["lambda_score"]}',
    ]
    return ' '.join(cmd_parts)


def start(gpu_workers=1, only_missing=True, preset_path=None):
    cmds = []
    missing = 0
    skipped = 0
    mode = "preset" if preset_path else "grid"

    if preset_path:
        cfg = load_preset_config(preset_path)
        model_groups = [
            (['PCA', 'IForest', 'OCSVM'], os.path.join(BASE_DIR, "main_baseline_pyod.py")),
            (['ECOD'], os.path.join(BASE_DIR, "main_baseline_pyod_ECOD.py")),
        ]
        for model_types, script_path in model_groups:
            for model_type in model_types:
                for dataname in datanames:
                    hp = resolve_job_hparams(cfg, dataname, model_type)
                    for seed in seeds:
                        if only_missing and has_result(
                            dataname,
                            model_type,
                            'none',
                            seed,
                            hp["query"],
                            hp["llm_type"],
                            hp["shuffle"],
                            hp["classifier"],
                            hp["lambda_score"],
                        ):
                            skipped += 1
                            continue
                        cmd = build_cmd(script_path, dataname, model_type, seed, hp)
                        cmds.append(cmd)
                        missing += 1
        print(f"Run mode: {mode}")
        print(f"Preset file: {preset_path}")
        print(f"Total pending jobs: {missing}")
        print(f"Skipped existing jobs: {skipped}")
        if missing == 0:
            return
        cuda_id = dict([(str(i), None) for i in range(gpu_workers)])
        run(cmds, cuda_id, gpu_workers)
        return

    # model_types = ['LOF']
    model_types = ['PCA', 'IForest', 'OCSVM']
    options = list(itertools.product(datanames, model_types, seeds, querys, llm_types, shuffles, classifiers, score_lambdas))
    # generate cmds of different experiments
    for dataname, model_type, seed, query, llm_type, shuffle, classifier, lambda_score in options:
        if only_missing and has_result(dataname, model_type, 'none', seed, query, llm_type, shuffle, classifier, lambda_score):
            skipped += 1
            continue

        hp = {
            "query": query,
            "llm_type": llm_type,
            "shuffle": shuffle,
            "classifier": classifier,
            "lambda_score": lambda_score,
        }
        cmd = build_cmd(os.path.join(BASE_DIR, "main_baseline_pyod.py"), dataname, model_type, seed, hp)
        cmds.append(cmd)
        missing += 1
    
    model_types = ['ECOD']
    options = list(itertools.product(datanames, model_types, seeds, querys, llm_types, shuffles, classifiers, score_lambdas))
    # generate cmds of different experiments
    for dataname, model_type, seed, query, llm_type, shuffle, classifier, lambda_score in options:
        if only_missing and has_result(dataname, model_type, 'none', seed, query, llm_type, shuffle, classifier, lambda_score):
            skipped += 1
            continue

        hp = {
            "query": query,
            "llm_type": llm_type,
            "shuffle": shuffle,
            "classifier": classifier,
            "lambda_score": lambda_score,
        }
        cmd = build_cmd(os.path.join(BASE_DIR, "main_baseline_pyod_ECOD.py"), dataname, model_type, seed, hp)
        cmds.append(cmd)
        missing += 1

    print(f"Run mode: {mode}")
    print(f"Total pending jobs: {missing}")
    print(f"Skipped existing jobs: {skipped}")

    if missing == 0:
        return

    cuda_id = dict([(str(i), None) for i in range(gpu_workers)])
    run(cmds, cuda_id, gpu_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1, help='number of concurrent workers')
    parser.add_argument('--rerun_all', action='store_true', help='rerun all jobs even if result exists')
    parser.add_argument('--preset', type=str, default='', help='optional preset json for per-model/per-dataset hyperparameters')
    args = parser.parse_args()
    preset = args.preset.strip() or None
    if preset and not os.path.isabs(preset):
        preset = os.path.join(BASE_DIR, preset)
    start(gpu_workers=args.workers, only_missing=(not args.rerun_all), preset_path=preset)
