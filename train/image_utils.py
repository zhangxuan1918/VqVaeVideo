import datetime

params = {
    'model_args': {
        'image_size': 256,
        'hidden_dim': 256,
        'num_resnet_blocks': 2,
        'smooth_l1_loss': False,
        'num_tokens': 8192,
        'codebook_dim': 512,
        'num_layers': 6,
        'normalization': None
    },
    'data_args': {
        'batch_size': 64,
        'root_dir': '/data/imagenet/ImageNet/train',
        'num_workers': 6
    },
    'train_args': {
        'num_steps': 250000, # we increase the batch size to reduce the steps
        'lr': 1e-3,
        'lr_decay': 0.98,
        'temp': 0.9,
        'temp_end': 1.0 / 16,
        'temp_anneal_rate': 1e-4,
        'kl_weight': 0.0,
        'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_image/2021-05-01',
        # 'checkpoint_path': '/opt/project/data/trained_image/2021-05-01/checkpoint136000.pth.tar',
        'checkpoint_path': None
    },
    'use_wandb': True
}
