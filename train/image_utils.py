import datetime

params = {
    'model_args': {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 256,
        'num_embeddings': 1024,
        'commitment_cost': 0.25,
        'decay': 0.99
    },
    'data_args': {
        'batch_size': 256,
        'root_dir': '/data/Doraemon/images/',
        'num_workers': 6
    },
    'train_args': {
        'num_steps': 250000, # we increase the batch size to reduce the steps
        'lr': 2e-3,
        # 'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        'folder_name': '/opt/project/data/trained_image/2021-04-27',
        'checkpoint_path': '/opt/project/data/trained_image/2021-04-27/checkpoint117000.pth.tar',
        'data_std': 1.0
    }
}
