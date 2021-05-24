import datetime

params = {
    'model_args': {
        'group_count': 7,
        'n_hid': 128,
        'n_blk_per_group': 1,
        'vocab_size': 8192 * 4,
        'n_init': 128,
        'input_channels': 3,
        'output_channels': 3,
        'commitment_cost': 0.25,
        'decay': 0.99
    },
    'data_args': {
        'batch_size': 32,
        'root_dir': '/data/Doraemon/images/',
        # 'root_dir': '/data/imagenet/ImageNet/train/',
        'num_workers': 6
    },
    'train_args': {
        'num_steps': 250000,  # we increase the batch size to reduce the steps
        'lr': 1e-4,
        'lr_decay': 0.98,
        'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_image/2021-05-01',
        # 'checkpoint_path': '/opt/project/data/trained_image/2021-05-01/checkpoint136000.pth.tar',
        'checkpoint_path': None
    },
    'use_wandb': False
}
