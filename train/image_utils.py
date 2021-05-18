import datetime

params = {
    'model_args': {
        'n_hid': 256,
        'n_blk_per_group': 2,
        'vocab_size': 8192,
        'requires_grad': True,
        'use_mixed_precision': True,
        'n_init': 128,
        'device': 'cuda:0'
    },
    'data_args': {
        'batch_size': 16,
        'root_dir': '/data/Doraemon/images/',
        'num_workers': 6
    },
    'train_args': {
        'num_steps': 250000, # we increase the batch size to reduce the steps
        'lr': 1e-4,
        'lr_decay': 0.999,

        'temp_start': 1.0,
        'temp_end': 1 / 16,
        'temp_anneal_rate': -2e-5,
        'kl_weight_start': 0.0,
        'kl_weight_end': 0.0,
        'kl_anneal_rate': 0.00132,

        'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_image/2021-05-01',
        # 'checkpoint_path': '/opt/project/data/trained_image/2021-05-01/checkpoint136000.pth.tar',
        'checkpoint_path': None
    },
    'use_wandb': False
}
