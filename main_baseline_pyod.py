import torch
import numpy as np
import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(BASE_DIR, ".cache"))
from scipy import io
import importlib
from DataSet.DataLoader import get_dataloader
from utils import aucPerformance, get_logger, F1Performance, sanitize_scores
import glob
import time
from sklearn.preprocessing import MinMaxScaler
from adbench.baseline.PyOD import PYOD
from adbench.baseline.DAGMM.run import DAGMM

DATA_DIR = os.path.join(BASE_DIR, 'Data')
npz_files = glob.glob(os.path.join(DATA_DIR, '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join(DATA_DIR, '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

def get_all_data(dataloader):
    all_data = []
    all_label = []
    for x, y in dataloader:
        all_data.append(x)
        all_label.append(y)
    return torch.cat(all_data, dim=0).numpy(), torch.cat(all_label, dim=0).numpy()

def evaluate(run, mse_rauc, mse_ap, mse_f1):
    mse_rauc[run], mse_ap[run] = aucPerformance(score, y_test)
    mse_f1[run] = F1Performance(score, y_test)

def import_function_from_file(folder_path, file_name, function_name):
    file_path = os.path.join(folder_path, file_name)
    
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    
    if spec is None:
        raise ImportError(f"cannot build from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    
    spec.loader.exec_module(module)
    
    if hasattr(module, function_name):
        return getattr(module, function_name)
    else:
        raise AttributeError(f"cannot find {function_name} from {file_name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--preprocess', type=str, default='none')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--query', type=int, default=1)
    parser.add_argument('--llm_type', type=str, default='gpt-4o')
    parser.add_argument('--shuffle', type=str, default='True')
    parser.add_argument('--classifier', type=str, default='rf')
    parser.add_argument('--lambda_score', type=float, default=1.0)
    args = parser.parse_args()
    if args.lambda_score <= 0:
        args.lambda_score = 1e-4


    dict_to_import = 'model_config_'+args.model_type
    module_name = 'configs'
    module = importlib.import_module(module_name)
    model_config = getattr(module, dict_to_import)

    model_config['preprocess'] = args.preprocess
    model_config['shuffle'] = args.shuffle
    model_config['random_seed'] = args.seed
    model_config['data_dir'] = DATA_DIR

    for result_dir in [
        "results_baseline", "models_baseline", "results_time_baseline",
        "hard_anomalies", "results_star", "models_star", "results_time_star"
    ]:
        os.makedirs(os.path.join(BASE_DIR, result_dir), exist_ok=True)

    if model_config['num_workers'] > 0:
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

    if args.dataname in npz_datanames:
        path = os.path.join(DATA_DIR, args.dataname + '.npz')
        data = np.load(path)
    elif args.dataname in mat_datanames:
        path = os.path.join(DATA_DIR, args.dataname + '.mat')
        data = io.loadmat(path)
    else:
        available = sorted(npz_datanames + mat_datanames)
        raise ValueError(f"Dataset '{args.dataname}' not found in {DATA_DIR}. Available: {available}")
    samples = data['X']
    model_config['dataset_name'] = args.dataname
    model_config['data_dim'] = samples.shape[-1]

    # Directly import AD algorithms from the existing toolkits like PyOD
    train_loader, test_loader = get_dataloader(model_config)
    X_train, y_train = get_all_data(train_loader)
    X_test, y_test = get_all_data(test_loader)

    result = []
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        if args.model_type == 'GOAD':
            from deepod.models.tabular import GOAD
            # model = GOAD(random_state=args.seed)
            # original 100
            model = GOAD(random_state=args.seed, epochs=5)
            start_time = time.time()
            model.fit(X_train, y_train)  # fit
            end_time = time.time()
            train_time = end_time - start_time
            start_time = time.time()
            score = model.decision_function(X_test)  # predict
            end_time = time.time()
            test_time = end_time - start_time
        elif args.model_type == 'NeuTraL':
            from deepod.models.tabular import NeuTraL
            # model = NeuTraL(random_state=args.seed)
            # original 100
            model = NeuTraL(random_state=args.seed, epochs=5)
            start_time = time.time()
            model.fit(X_train, y_train)  # fit
            end_time = time.time()
            train_time = end_time - start_time
            start_time = time.time()
            score = model.decision_function(X_test)  # predict
            end_time = time.time()
            test_time = end_time - start_time
        elif args.model_type == 'ICL':
            from deepod.models.tabular import ICL
            # model = ICL(random_state=args.seed)
            # original 100
            model = ICL(random_state=args.seed, epochs=5)
            start_time = time.time()
            model.fit(X_train, y_train)  # fit
            end_time = time.time()
            train_time = end_time - start_time
            start_time = time.time()
            score = model.decision_function(X_test)  # predict
            end_time = time.time()
            test_time = end_time - start_time
        elif args.model_type == 'DAGMM':
            # model = DAGMM(seed=args.seed)
            # original 200
            model = DAGMM(seed=args.seed, num_epochs=5)
            start_time = time.time()
            model.fit(X_train, y_train)  # fit
            end_time = time.time()
            train_time = end_time - start_time
            start_time = time.time()
            score = model.predict_score(X_train, X_test)  # predict
            end_time = time.time()
            test_time = end_time - start_time
        else:
            model = PYOD(seed=args.seed, model_name=args.model_type)  # initialization
            start_time = time.time()
            model.fit(X_train, y_train)  # fit
            end_time = time.time()
            train_time = end_time - start_time
            start_time = time.time()
            score = model.predict_score(X_test)  # predict
            score = sanitize_scores(score)
            end_time = time.time()
            test_time = end_time - start_time
        score = sanitize_scores(score)
        evaluate(i, mse_rauc, mse_ap, mse_f1)
    mean_mse_auc , mean_mse_pr , mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)

    results_dict = {'AUC-ROC':mean_mse_auc, 'AUC-PR':mean_mse_pr, 'f1':mean_mse_f1}
    legacy_tag = f"{args.dataname}_{args.model_type}_{args.preprocess}_{args.seed}_{args.query}_{args.llm_type}_{args.shuffle}_{args.classifier}"
    tag = f"{legacy_tag}_{args.lambda_score:.6g}"
    np.save(open(os.path.join(BASE_DIR, f'results_baseline/{tag}.npy'),'wb'), results_dict)
    np.save(open(os.path.join(BASE_DIR, f'results_baseline/{legacy_tag}.npy'),'wb'), results_dict)
    torch.save(model, os.path.join(BASE_DIR, f"models_baseline/{tag}.pth"))
    torch.save(model, os.path.join(BASE_DIR, f"models_baseline/{legacy_tag}.pth"))
    np.save(open(os.path.join(BASE_DIR, f'results_time_baseline/{tag}_train.npy'),'wb'), train_time)
    np.save(open(os.path.join(BASE_DIR, f'results_time_baseline/{tag}_test.npy'),'wb'), test_time)
    np.save(open(os.path.join(BASE_DIR, f'results_time_baseline/{legacy_tag}_train.npy'),'wb'), train_time)
    np.save(open(os.path.join(BASE_DIR, f'results_time_baseline/{legacy_tag}_test.npy'),'wb'), test_time)

    # phase 2
    start_time = time.time()

    folder = os.path.join(BASE_DIR, f"answer/code/{args.llm_type}")
    file = f"{args.model_type}_{args.query}.py"
    function = "generate_hard_anomalies"
    try:
        generate_hard_anomalies = import_function_from_file(folder, file, function)
    except (ImportError, AttributeError) as e:
        print(f"error: {e}")

    # if os.path.exists(f'./hard_anomalies/{args.dataname}_{args.model_type}_{args.preprocess}_{args.seed}_{args.query}_{args.llm_type}_{args.shuffle}_{args.classifier}.npy'):
    #     hard_anomalies = np.load(
    #         f'./hard_anomalies/{args.dataname}_{args.model_type}_{args.preprocess}_{args.seed}_{args.query}_{args.llm_type}_{args.shuffle}_{args.classifier}.npy',
    #         allow_pickle=True
    #     )
    # else:
    #     hard_anomalies = generate_hard_anomalies(n_samples=int(X_train.shape[0]*0.1), model=model, X_train=X_train)
    #     np.save(open(f'./hard_anomalies/{args.dataname}_{args.model_type}_{args.preprocess}_{args.seed}_{args.query}_{args.llm_type}_{args.shuffle}_{args.classifier}.npy','wb'), hard_anomalies)
    hard_anomalies = generate_hard_anomalies(n_samples=int(X_train.shape[0]*0.1), model=model, X_train=X_train)
    np.save(open(os.path.join(BASE_DIR, f'hard_anomalies/{tag}.npy'),'wb'), hard_anomalies)
    np.save(open(os.path.join(BASE_DIR, f'hard_anomalies/{legacy_tag}.npy'),'wb'), hard_anomalies)
    
    
    
    hard_labels = np.ones(hard_anomalies.shape[0])

    X_train_aug = np.concatenate([X_train, hard_anomalies], axis=0)
    y_train_aug = np.concatenate([y_train, hard_labels])

    result = []
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        if args.classifier == 'mlp':
            from MLP_predictor import Wrapper
            model_1 = Wrapper(input_dim=X_train_aug.shape[1], hidden_dim=256, lr=1e-2, n_epochs=200)

        elif args.classifier == 'rf':
            from RF_predictor import RandomForestWrapper
            # model_1  = RandomForestWrapper(n_estimators=200, max_depth=5, random_state=42)
            model_1  = RandomForestWrapper(random_state=args.seed)

        elif args.classifier == 'catboost':
            from Catboost_predictor import CatBoostWrapper
            model_1 = CatBoostWrapper(use_gpu=True, random_state=args.seed)
        
        elif args.classifier == 'svm':
            from SVM_predictor import SVMWrapper
            model_1  = SVMWrapper(random_state=args.seed)

        model_1.fit(X_train_aug, y_train_aug)

        end_time = time.time()
        train_time = end_time - start_time

        start_time = time.time()
        # from binary classifier
        train_score = model_1.predict_score(X_train)
        test_score = model_1.predict_score(X_test)
        train_score = sanitize_scores(train_score)
        test_score = sanitize_scores(test_score)
        # fit normalizer using train score
        scaler = MinMaxScaler()
        scaler.fit(train_score.reshape(-1,1))
        # normalize test score using the fitted normalizer
        test_score_norm = scaler.transform(test_score.reshape(-1,1)).flatten()

        # from original detector
        train_score_original = model.predict_score(X_train)
        test_score_original = model.predict_score(X_test)
        train_score_original = sanitize_scores(train_score_original)
        test_score_original = sanitize_scores(test_score_original)
        # fit normalizer using train score
        scaler = MinMaxScaler()
        scaler.fit(train_score_original.reshape(-1,1))
        # normalize test score using the fitted normalizer
        test_score_original_norm = scaler.transform(test_score_original.reshape(-1,1)).flatten()

        # ensemble
        score = args.lambda_score * test_score_norm + test_score_original_norm

        end_time = time.time()
        test_time = end_time - start_time
        score = sanitize_scores(score)
        evaluate(i, mse_rauc, mse_ap, mse_f1)

    mean_mse_auc , mean_mse_pr , mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)

    print('##########################################################################')
    print("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f"
          % (mean_mse_auc, mean_mse_pr))
    print("mse: average f1: %.4f" % (mean_mse_f1))
    ##

    results_dict = {'AUC-ROC':mean_mse_auc, 'AUC-PR':mean_mse_pr, 'f1':mean_mse_f1, 'lambda_score': args.lambda_score}
    np.save(open(os.path.join(BASE_DIR, f'results_star/{tag}.npy'),'wb'), results_dict)
    np.save(open(os.path.join(BASE_DIR, f'results_star/{legacy_tag}.npy'),'wb'), results_dict)
    torch.save(model_1, os.path.join(BASE_DIR, f"models_star/{tag}.pth"))
    torch.save(model_1, os.path.join(BASE_DIR, f"models_star/{legacy_tag}.pth"))
    np.save(open(os.path.join(BASE_DIR, f'results_time_star/{tag}_train.npy'),'wb'), train_time)
    np.save(open(os.path.join(BASE_DIR, f'results_time_star/{tag}_test.npy'),'wb'), test_time)
    np.save(open(os.path.join(BASE_DIR, f'results_time_star/{legacy_tag}_train.npy'),'wb'), train_time)
    np.save(open(os.path.join(BASE_DIR, f'results_time_star/{legacy_tag}_test.npy'),'wb'), test_time)
