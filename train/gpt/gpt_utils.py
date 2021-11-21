import datetime

params = {
    'model_args': {
        'vocab_size': 8192,
        'max_seq_length': 256,
        'cond_seq_length': 16,
        'num_layers': 6,
        'num_heads': 8,
        'embed_dim': 512,
        'embed_dropout_prob': 0.1,
        'mlp_dropout_prob': 0.1,
        'attn_dropout_prob': 0.1,
        'pretrained_visual_embed_path': 'visual_embed_weights_2021-05-25.npz'
    },
    'data_args': {
        'batch_size': 8,
        'root_dir': '/data/Doraemon/np_arrays/256x256',
        'num_workers': 0,
        'max_seq_length': 16,
        'padding_file': 'black_images_code_2021-05-25.npz'
    },
    'train_args': {
        'num_steps': 80000,  # we increase the batch size to reduce the steps
        'lr': 1e-5,
        'lr_decay': 0.999,
        'folder_name': '/opt/project/data/trained_gpt/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        # 'folder_name': '/opt/project/data/trained_image/2021-05-01',
        # 'checkpoint_path': '/opt/project/data/trained_image/2021-05-01/checkpoint136000.pth.tar',
        'checkpoint_path': None
    },
    'use_wandb': False
}
