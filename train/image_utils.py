import datetime

params = {
    'model_args': {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 512,
        'embedding_mul': 1, # used only by video
        'num_embeddings': 8192,
        'commitment_cost': 0.25,
        'decay': 0.99
    },
    'data_args': {
        'batch_size': 64,
        'root_dir': '/data/Doraemon/images/',
        'num_workers': 6
    },
    'train_args': {
        'num_steps': 250000, # we increase the batch size to reduce the steps
        'lr': 1e-3,
        'lr_decay': 0.98,
        # 'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        'folder_name': '/opt/project/data/trained_image/2021-05-01',
        'checkpoint_path': '/opt/project/data/trained_image/2021-05-01/checkpoint136000.pth.tar',
        # 'checkpoint_path': None
    }
}
