import datetime

params = {
    'model_args': {
        'n_hid': 128,
        'n_blk_per_group': 2,
        'input_channels': 128,
        'vocab_size': 8192,
        'requires_grad': True,
        'use_mixed_precision': False,
        'n_init': 128,
        'output_channels': 128,
        'device': 'cuda:0'
    },
    'data_args': {
        'batch_size': 16,
        'root_dir': '/data/Doraemon/np_arrays/',
        'num_workers': 6
    },
    'train_args': {
        'num_steps': 250000, # we increase the batch size to reduce the steps
        'lr': 1e-3,
        'lr_decay': 0.98,
        'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_image/2021-05-01',
        # 'checkpoint_path': '/opt/project/data/trained_image/2021-05-01/checkpoint136000.pth.tar',
        'checkpoint_path': None
    }
}
