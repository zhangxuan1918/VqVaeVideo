import datetime

params = {
    'model_args': {
        'group_count': 4,
        'n_hid': 64,
        'n_blk_per_group': 1,
        'vocab_size': 8192,
        'n_init': 512,
        'input_channels': 256,
        'output_channels': 256,
        'downsample': False, # no downsampling as we compress in time dim
        'upsample': False,
        'commitment_cost': 0.25,
        'decay': 0.99
    },
    'data_args': {
        'batch_size': 128,
        'root_dir': '/data/breaking_bad/np_arrays/256x256',
        'num_workers': 0,
        'max_seq_length': 256, # max frames for each video input
        'padding_file': 'black_images_code_2021-05-25.npy.gz',
    },
    'train_args': {
        'num_steps': 80000,  # we increase the batch size to reduce the steps
        'lr': 1e-4,
        'lr_decay': 0.98,
        'folder_name': '/opt/project/data/trained_code/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_code/2021-05-01',
        # 'checkpoint_path': '/opt/project/data/trained_code/2021-05-01/checkpoint136000.pth.tar',
        'checkpoint_path': None
    },
    'use_wandb': False
}
