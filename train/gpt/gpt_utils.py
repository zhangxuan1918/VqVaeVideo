import datetime

params = {
    'model_args': {
        'vocab_size': 8192,
        'max_seq_length': 512,
        'num_layers': 6,
        'num_heads': 8,
        'embed_dim': 512,
        'embed_dropout_prob': 0.1,
        'mlp_dropout_prob': 0.1,
        'attn_dropout_prob': 0.1
    },
    'data_args': {
        'batch_size': 32,
        'root_dir': '/data/Doraemon/images/',
        # 'root_dir': '/data/imagenet/ImageNet/train/',
        'num_workers': 0
    },
    'train_args': {
        'num_steps': 80000,  # we increase the batch size to reduce the steps
        'lr': 1e-4,
        'lr_decay': 0.98,
        'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_image/2021-05-01',
        # 'checkpoint_path': '/opt/project/data/trained_image/2021-05-01/checkpoint136000.pth.tar',
        'checkpoint_path': None
    },
    'use_wandb': False
}
