import datetime

params = {
    'model_args': {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'decay': 0.99
    },
    'data_args': {
        'batch_size': 1024
    },
    'train_args': {
        'num_steps': 3750, # we increase the batch size to reduce the steps
        'lr': 1e-3,
        'folder_name': '/opt/project/data/trained_image/' + datetime.datetime.today().strftime('%Y-%m-%d'),
        'data_std': 0.063287
    }
}
