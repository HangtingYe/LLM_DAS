from torch.utils.data import DataLoader
from DataSet.MyDataset import CsvDataset, MatDataset, NpzDataset
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_dataloader(model_config: dict):
    dataset_name = model_config['dataset_name']
    data_dir = model_config['data_dir']
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
    data_dir = os.path.abspath(data_dir)

    npz_path = os.path.join(data_dir, dataset_name + '.npz')
    mat_path = os.path.join(data_dir, dataset_name + '.mat')

    if os.path.exists(npz_path):
        train_set = NpzDataset(dataset_name, model_config['data_dim'], data_dir, model_config['preprocess'], model_config, mode='train')
        test_set = NpzDataset(dataset_name, model_config['data_dim'], data_dir, model_config['preprocess'], model_config, mode='eval')

    elif os.path.exists(mat_path):
        train_set = MatDataset(dataset_name, model_config['data_dim'], data_dir, model_config['preprocess'], model_config, mode='train')
        test_set = MatDataset(dataset_name, model_config['data_dim'], data_dir, model_config['preprocess'], model_config, mode='eval')
        
    else:
        train_set = CsvDataset(dataset_name, model_config['data_dim'], data_dir, model_config['preprocess'], model_config, mode='train')
        test_set = CsvDataset(dataset_name, model_config['data_dim'], data_dir, model_config['preprocess'], model_config, mode='eval')

    train_loader = DataLoader(train_set,
                              batch_size=model_config['batch_size'],
                              num_workers=model_config['num_workers'],
                              shuffle=False,
                              )
    test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)
    return train_loader, test_loader
